"""Runtime-mode helpers: flight phases, recovery mode, and backward repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PositionColumns = ("east_m", "north_m", "up_m")


@dataclass(frozen=True)
class RecoveryDecision:
    """Configuration multipliers used while the tracker is in recovery mode."""

    active: bool
    reason: str
    candidate_pool_multiplier: float = 1.0
    rf_anchor_weight_multiplier: float = 1.0
    track_switch_penalty_multiplier: float = 1.0
    gate_probability: float | None = None


@dataclass
class RecoveryModeController:
    """Simple state machine for association-collapse recovery."""

    nis_threshold: float = 25.0
    low_confidence_threshold: float = 0.2
    miss_streak_threshold: int = 3
    recovery_frames: int = 5
    remaining_frames: int = 0

    def update(
        self,
        *,
        nis: float | None = None,
        association_confidence: float | None = None,
        miss_streak: int = 0,
        track_switched: bool = False,
    ) -> RecoveryDecision:
        triggers = []
        if nis is not None and np.isfinite(float(nis)) and float(nis) > float(self.nis_threshold):
            triggers.append("high_nis")
        if association_confidence is not None and float(association_confidence) < float(self.low_confidence_threshold):
            triggers.append("low_confidence")
        if int(miss_streak) >= int(self.miss_streak_threshold):
            triggers.append("miss_streak")
        if track_switched:
            triggers.append("track_switch")
        if triggers:
            self.remaining_frames = int(self.recovery_frames)
        elif self.remaining_frames > 0:
            self.remaining_frames -= 1
        active = self.remaining_frames > 0
        return RecoveryDecision(
            active=active,
            reason="+".join(triggers) if triggers else ("cooldown" if active else "nominal"),
            candidate_pool_multiplier=3.0 if active else 1.0,
            rf_anchor_weight_multiplier=2.0 if active else 1.0,
            track_switch_penalty_multiplier=0.25 if active else 1.0,
            gate_probability=0.999 if active else None,
        )


def segment_flight_phases(frame: pd.DataFrame) -> pd.Series:
    """Assign coarse test-time flight phases from positions and timestamps."""

    if frame.empty:
        return pd.Series(dtype=str)
    ordered = frame.sort_values("time_s") if "time_s" in frame.columns else frame.copy()
    times = pd.to_numeric(ordered.get("time_s", pd.Series(range(len(ordered)))), errors="coerce").to_numpy(dtype=float)
    positions = ordered.loc[:, [c for c in PositionColumns if c in ordered.columns]].to_numpy(dtype=float)
    if positions.shape[1] < 2 or len(ordered) < 3:
        return pd.Series(["unknown"] * len(ordered), index=ordered.index)
    dt = np.diff(times, prepend=times[0])
    dt = np.where(dt > 1e-6, dt, np.nan)
    velocity = np.vstack([np.zeros(positions.shape[1]), np.diff(positions, axis=0)]) / dt[:, None]
    speed = np.linalg.norm(np.nan_to_num(velocity[:, :2]), axis=1)
    altitude = positions[:, 2] if positions.shape[1] >= 3 else np.zeros(len(positions))
    speed_hi = np.nanpercentile(speed, 75) if np.isfinite(speed).any() else 0.0
    alt_lo = np.nanpercentile(altitude, 20) if np.isfinite(altitude).any() else 0.0
    phase = np.full(len(ordered), "cruise", dtype=object)
    phase[speed < max(speed_hi * 0.25, 1.0)] = "slow"
    phase[altitude <= alt_lo] = "low-altitude"
    if len(phase) >= 4:
        phase[: max(1, len(phase) // 10)] = "takeoff-or-start"
        phase[-max(1, len(phase) // 10) :] = "landing-or-end"
    headings = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    turn_rate = np.abs(np.diff(headings, prepend=headings[0]))
    phase[turn_rate > np.nanpercentile(turn_rate, 90)] = "turn"
    return pd.Series(phase, index=ordered.index).reindex(frame.index)


def backward_repair_associations(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    max_gap_s: float = 10.0,
    max_repair_distance_m: float = 200.0,
) -> pd.DataFrame:
    """Repair suspicious selected-radar gaps using a backward anchor pass.

    The function interpolates between selected anchors and fills missing radar
    frames with the candidate closest to the interpolated path.  It is intended
    for offline/fixed-lag diagnostics, not strict causal tracking.
    """

    if selected.empty or candidates.empty:
        return selected.copy()
    if "sequence_id" not in selected.columns or "sequence_id" not in candidates.columns:
        return _backward_repair_one_sequence(
            selected,
            candidates,
            max_gap_s=max_gap_s,
            max_repair_distance_m=max_repair_distance_m,
        )

    selected_rows = selected.copy()
    candidate_rows = candidates.copy()
    selected_rows["_sequence_key"] = _sequence_keys(selected_rows["sequence_id"])
    candidate_rows["_sequence_key"] = _sequence_keys(candidate_rows["sequence_id"])
    repaired_parts: list[pd.DataFrame] = []
    for sequence_key in pd.unique(selected_rows["_sequence_key"]):
        selected_mask = _sequence_mask(selected_rows["_sequence_key"], sequence_key)
        candidate_mask = _sequence_mask(candidate_rows["_sequence_key"], sequence_key)
        sequence_selected = selected_rows.loc[selected_mask].drop(columns="_sequence_key")
        sequence_candidates = candidate_rows.loc[candidate_mask].drop(columns="_sequence_key")
        if sequence_candidates.empty:
            repaired_parts.append(sequence_selected)
            continue
        repaired_parts.append(
            _backward_repair_one_sequence(
                sequence_selected,
                sequence_candidates,
                max_gap_s=max_gap_s,
                max_repair_distance_m=max_repair_distance_m,
            )
        )
    return (
        pd.concat(repaired_parts, ignore_index=True, sort=False)
        .sort_values(["sequence_id", "time_s"], kind="mergesort")
        .reset_index(drop=True)
    )


def _backward_repair_one_sequence(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    max_gap_s: float,
    max_repair_distance_m: float,
) -> pd.DataFrame:
    selected = selected.sort_values("time_s").reset_index(drop=True)
    repaired = [row.copy() for _, row in selected.iterrows()]
    frame_key_column = _frame_key_column(candidates)
    candidate_groups = _frame_groups(candidates, frame_key_column=frame_key_column)
    selected_keys = {
        key
        for _, row in selected.iterrows()
        if (key := _row_key(row, frame_key_column=frame_key_column)) is not None
    }
    for left, right in zip(selected.iloc[:-1].itertuples(index=False), selected.iloc[1:].itertuples(index=False)):
        left_time = float(left.time_s)
        right_time = float(right.time_s)
        gap_s = right_time - left_time
        if gap_s <= 0.0 or gap_s > float(max_gap_s):
            continue
        left_pos = np.array([left.east_m, left.north_m, left.up_m], dtype=float)
        right_pos = np.array([right.east_m, right.north_m, right.up_m], dtype=float)
        if not np.isfinite(left_pos).all() or not np.isfinite(right_pos).all():
            continue
        for key, frame in candidate_groups:
            if key in selected_keys:
                continue
            time_s = float(
                pd.to_numeric(frame["time_s"], errors="coerce").median()
            )
            if not np.isfinite(time_s) or not left_time < time_s < right_time:
                continue
            alpha = (time_s - left_time) / gap_s
            target = (1.0 - alpha) * left_pos + alpha * right_pos
            positions = (
                frame.loc[:, PositionColumns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
            finite = np.isfinite(positions).all(axis=1)
            if not finite.any():
                continue
            distances = np.full(len(frame), np.inf, dtype=float)
            distances[finite] = np.linalg.norm(positions[finite] - target.reshape(1, 3), axis=1)
            best_idx = int(np.argmin(distances))
            if float(distances[best_idx]) <= float(max_repair_distance_m):
                row = frame.iloc[best_idx].copy()
                row["association_mode"] = "backward-repair"
                row["association_score"] = float(distances[best_idx])
                row["association_repaired"] = True
                repaired.append(row)
                selected_keys.add(key)
    return pd.DataFrame(repaired).sort_values("time_s").reset_index(drop=True)


def _sequence_keys(values: pd.Series) -> pd.Series:
    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    return keys.where(keys.notna() & keys.ne(""))


def _sequence_mask(keys: pd.Series, value: object) -> pd.Series:
    if pd.isna(value):
        return keys.isna()
    return keys.eq(value).fillna(False)


def _frame_key_column(frame: pd.DataFrame) -> str:
    if "frame_index" in frame.columns:
        values = pd.to_numeric(frame["frame_index"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).all():
            return "frame_index"
    if "time_s" in frame.columns:
        return "time_s"
    if "frame_index" in frame.columns:
        return "frame_index"
    raise KeyError("radar candidates must contain frame_index or time_s")


def _frame_groups(
    frame: pd.DataFrame,
    *,
    frame_key_column: str | None = None,
) -> list[tuple[object, pd.DataFrame]]:
    key_column = frame_key_column or _frame_key_column(frame)
    work = frame.copy()
    values = pd.to_numeric(work[key_column], errors="coerce")
    work["_frame_key"] = values if key_column == "frame_index" else values.round(9)
    return [
        (key, rows.drop(columns="_frame_key").copy())
        for key, rows in work.groupby("_frame_key", sort=True)
    ]


def _row_key(
    row: pd.Series | object,
    *,
    frame_key_column: str | None = None,
) -> object | None:
    if isinstance(row, pd.Series):
        get_value = row.get
    else:
        get_value = lambda name, default=None: getattr(row, name, default)
    key_column = frame_key_column
    if key_column is None:
        frame_index = _finite_number(get_value("frame_index", np.nan))
        key_column = "frame_index" if frame_index is not None else "time_s"
    value = _finite_number(get_value(key_column, np.nan))
    if value is None:
        return None
    return value if key_column == "frame_index" else round(value, 9)


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None
