"""Basic MMUAD tracking backend inspired by RaFT-UAV++."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame, TruthFrame


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for the first MMUAD backend."""

    acceleration_std_mps2: float = 8.0
    primary_covariance_scale: float = 1.0
    secondary_covariance_scale: float = 25.0
    soft_anchor_cap_m: float = 2.0
    first_selected_bootstrap: bool = True
    source_priority: tuple[str, ...] = ("radar", "lidar", "lidar-cluster", "candidate")


@dataclass(frozen=True)
class TrackerOutput:
    estimates: pd.DataFrame
    metrics: dict[str, Any]
    selected_tracklets: pd.DataFrame


_CANDIDATE_ROW_ID = "_candidate_row_id"


def run_mmuad_tracker(
    candidates: CandidateFrame,
    truth: TruthFrame | None = None,
    *,
    config: TrackerConfig | None = None,
) -> TrackerOutput:
    """Run a single-UAV tracking-by-detection baseline on normalized candidates.

    The tracker is intentionally modest.  It selects one high-quality tracklet
    path using candidate metadata and motion smoothness, initializes at the
    first selected measurement, applies full updates for selected candidates,
    and applies capped soft-anchor updates for other candidates.
    """

    config = config or TrackerConfig()
    rows = candidates.rows.copy()
    candidates.validate()
    rows = _candidate_rows_with_optional_defaults(rows)
    if rows.empty:
        empty = pd.DataFrame()
        return TrackerOutput(empty, {"count": 0}, empty)

    estimates_by_sequence: list[pd.DataFrame] = []
    selected_by_sequence: list[pd.DataFrame] = []
    metrics_by_sequence: dict[str, Any] = {}
    truth_rows = truth.rows if truth is not None else None
    for sequence_id, sequence_candidates in rows.groupby("sequence_id", sort=True):
        sequence_candidates = sequence_candidates.copy()
        sequence_candidates[_CANDIDATE_ROW_ID] = np.arange(len(sequence_candidates), dtype=int)
        selected = select_tracklet_path(sequence_candidates, config=config)
        selected_by_sequence.append(
            selected.drop(columns=[_CANDIDATE_ROW_ID], errors="ignore").assign(
                sequence_id=sequence_id
            )
        )
        sequence_truth = None
        if truth_rows is not None:
            sequence_truth = truth_rows.loc[truth_rows["sequence_id"] == sequence_id]
        estimates = _run_sequence_filter(
            sequence_candidates,
            selected,
            sequence_truth=sequence_truth,
            config=config,
        )
        estimates_by_sequence.append(estimates.assign(sequence_id=sequence_id))
        metrics_by_sequence[str(sequence_id)] = compute_metrics(estimates, sequence_truth)

    estimates_all = (
        pd.concat(estimates_by_sequence, ignore_index=True)
        if estimates_by_sequence
        else pd.DataFrame()
    )
    selected_all = (
        pd.concat(selected_by_sequence, ignore_index=True)
        if selected_by_sequence
        else pd.DataFrame()
    )
    metrics = {
        "sequences": metrics_by_sequence,
        "pooled": compute_metrics(estimates_all, truth_rows),
    }
    return TrackerOutput(estimates_all, metrics, selected_all)


def _candidate_rows_with_optional_defaults(rows: pd.DataFrame) -> pd.DataFrame:
    """Return candidate rows with optional tracker columns present.

    ``CandidateFrame.validate`` intentionally requires only the columns needed
    to define a candidate position. The tracker can also use optional metadata
    such as stable track IDs, confidence scores, and per-row covariance hints.
    Fill the same defaults used by the CSV normalization path so hand-built
    ``CandidateFrame`` inputs with the minimal valid schema do not fail later
    with ``KeyError`` during path selection or filtering.
    """

    out = rows.copy()
    if "track_id" not in out.columns:
        out["track_id"] = np.nan
    if "confidence" not in out.columns:
        out["confidence"] = 1.0
    if "std_xy_m" not in out.columns:
        out["std_xy_m"] = 10.0
    if "std_z_m" not in out.columns:
        out["std_z_m"] = out["std_xy_m"]
    return out


def select_tracklet_path(candidates: pd.DataFrame, *, config: TrackerConfig) -> pd.DataFrame:
    """Select a single globally plausible candidate tracklet path.

    If stable track IDs are present, score each source/track_id group.  If not,
    fall back to a greedy nearest-neighbor path over highest-confidence rows.
    """

    frame = _candidate_rows_with_optional_defaults(candidates).sort_values("time_s")
    frame = frame.loc[_finite_position_mask(frame)].copy()
    if frame.empty:
        return candidates.iloc[0:0].copy()
    non_null_track = frame.loc[frame["track_id"].notna()].copy()
    if not non_null_track.empty:
        rows: list[dict[str, object]] = []
        for (source, track_id), group in non_null_track.groupby(["source", "track_id"], sort=True):
            group = group.sort_values("time_s")
            score = _tracklet_score(group, config=config)
            rows.append(
                {
                    "source": source,
                    "track_id": track_id,
                    "score": score,
                    "rows": len(group),
                }
            )
        ranked = pd.DataFrame.from_records(rows).sort_values(
            ["score", "rows"],
            ascending=[True, False],
        )
        best = ranked.iloc[0]
        selected = non_null_track.loc[
            (non_null_track["source"] == best["source"])
            & (non_null_track["track_id"] == best["track_id"])
        ].copy()
        selected["selected_path_rank"] = 0
        selected["selected_path_score"] = float(best["score"])
        return selected.sort_values("time_s").reset_index(drop=True)
    return _greedy_path(frame, config=config)


def _tracklet_score(group: pd.DataFrame, *, config: TrackerConfig) -> float:
    confidence = (
        pd.to_numeric(group.get("confidence", 1.0), errors="coerce")
        .fillna(1.0)
        .to_numpy(float)
    )
    priority = _source_priority(str(group["source"].iloc[0]), config=config)
    xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
    t = group["time_s"].to_numpy(float)
    speed_penalty = 0.0
    if len(group) >= 2:
        dt = np.maximum(np.diff(t), 1.0e-3)
        speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
        speed_penalty = float(np.nanmedian(speed) / 25.0 + np.nanpercentile(speed, 95.0) / 80.0)
    return float(priority + speed_penalty - 0.05 * len(group) - 0.25 * np.nanmean(confidence))


def _source_priority(source: str, *, config: TrackerConfig) -> float:
    source_lower = source.lower()
    best_idx: int | None = None
    best_length = -1
    for idx, name in enumerate(config.source_priority):
        name_lower = name.lower()
        if source_lower.startswith(name_lower) and len(name_lower) > best_length:
            best_idx = idx
            best_length = len(name_lower)
    if best_idx is not None:
        return float(best_idx)
    return float(len(config.source_priority))


def _greedy_path(frame: pd.DataFrame, *, config: TrackerConfig) -> pd.DataFrame:
    del config
    ranked = frame.sort_values(["time_s", "confidence"], ascending=[True, False]).copy()
    chosen_rows = []
    last_xyz: np.ndarray | None = None
    last_time: float | None = None
    for time_s, group in ranked.groupby("time_s", sort=True):
        group = group.loc[_finite_position_mask(group)].copy()
        if group.empty:
            continue
        if last_xyz is None or last_time is None:
            chosen = group.sort_values("confidence", ascending=False).iloc[0]
        else:
            xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
            dt = max(float(time_s) - float(last_time), 1.0)
            distances = np.linalg.norm(xyz - last_xyz, axis=1) / dt
            chosen = group.iloc[int(np.argmin(distances))]
        last_xyz = chosen[["x_m", "y_m", "z_m"]].to_numpy(float)
        last_time = float(chosen["time_s"])
        chosen_rows.append(chosen)
    if not chosen_rows:
        return frame.iloc[0:0].copy()
    selected = pd.DataFrame(chosen_rows).reset_index(drop=True)
    selected["selected_path_rank"] = 0
    selected["selected_path_score"] = 0.0
    return selected


def _run_sequence_filter(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    sequence_truth: pd.DataFrame | None,
    config: TrackerConfig,
) -> pd.DataFrame:
    selected = selected.loc[_finite_position_mask(selected)].copy()
    if selected.empty:
        return pd.DataFrame()
    selected_keys = set(_candidate_keys(selected))
    events = (
        candidates.loc[_finite_position_mask(candidates)]
        .sort_values("time_s")
        .reset_index(drop=True)
    )
    if events.empty:
        return pd.DataFrame()
    bootstrap = selected.sort_values("time_s").iloc[0]
    filt = _ConstantVelocityFilter(
        acceleration_std_mps2=config.acceleration_std_mps2,
        initial_time_s=float(bootstrap["time_s"]),
        initial_position=bootstrap[["x_m", "y_m", "z_m"]].to_numpy(float),
    )
    estimate_rows: list[dict[str, object]] = []
    for _, row in events.iterrows():
        time_s = float(row["time_s"])
        if config.first_selected_bootstrap and time_s < float(bootstrap["time_s"]):
            continue
        filt.predict(time_s)
        key = _candidate_key(row)
        is_selected = key in selected_keys
        z = row[["x_m", "y_m", "z_m"]].to_numpy(float)
        std_xy = _positive_float(row.get("std_xy_m", 10.0), default=10.0)
        std_z = _positive_float(row.get("std_z_m", std_xy), default=std_xy)
        covariance = np.diag([std_xy**2, std_xy**2, std_z**2])
        if is_selected:
            action = "selected_update"
            filt.update(z, covariance * config.primary_covariance_scale)
        else:
            action = "soft_anchor"
            predicted = filt.state[:3].copy()
            innovation = z - predicted
            horizontal_norm = float(np.linalg.norm(innovation[:2]))
            if horizontal_norm > config.soft_anchor_cap_m > 0:
                innovation[:2] *= config.soft_anchor_cap_m / horizontal_norm
            capped_z = predicted + innovation
            filt.update(capped_z, covariance * config.secondary_covariance_scale)
        state = filt.state.copy()
        record = {
            "time_s": time_s,
            "source": row.get("source"),
            "track_id": row.get("track_id"),
            "class_name": row.get("class_name"),
            "update_action": action,
            "selected_path_update": bool(is_selected),
            "state_x_m": state[0],
            "state_y_m": state[1],
            "state_z_m": state[2],
            "v_x_mps": state[3],
            "v_y_mps": state[4],
            "v_z_mps": state[5],
        }
        estimate_rows.append(record)
    estimates = pd.DataFrame.from_records(estimate_rows)
    if sequence_truth is not None and not sequence_truth.empty and not estimates.empty:
        estimates = add_truth_errors(estimates, sequence_truth)
    return estimates


def _candidate_keys(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    return [_candidate_key(row) for _, row in frame.iterrows()]


def _candidate_key(row: pd.Series) -> tuple[object, ...]:
    if _CANDIDATE_ROW_ID in row.index and pd.notna(row[_CANDIDATE_ROW_ID]):
        return ("row", int(row[_CANDIDATE_ROW_ID]))
    return ("fields", float(row["time_s"]), str(row.get("source", "")), str(row.get("track_id", "")))


def _finite_position_mask(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=bool)
    positions = frame[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce")
    times = pd.to_numeric(frame["time_s"], errors="coerce")
    return (np.isfinite(positions).all(axis=1) & np.isfinite(times)).to_numpy(dtype=bool)


def _positive_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) and number > 0.0 else float(default)


class _ConstantVelocityFilter:
    def __init__(
        self,
        *,
        acceleration_std_mps2: float,
        initial_time_s: float,
        initial_position: np.ndarray,
    ) -> None:
        self.acceleration_std_mps2 = float(acceleration_std_mps2)
        self.time_s = float(initial_time_s)
        self.state = np.zeros(6, dtype=float)
        self.state[:3] = initial_position.astype(float)
        self.covariance = np.diag([25.0, 25.0, 25.0, 100.0, 100.0, 100.0])

    def predict(self, time_s: float) -> None:
        dt = float(time_s) - self.time_s
        if dt <= 0.0:
            self.time_s = float(time_s)
            return
        f = np.eye(6)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt
        q_pos = 0.25 * dt**4 * self.acceleration_std_mps2**2
        q_vel = dt**2 * self.acceleration_std_mps2**2
        q_cross = 0.5 * dt**3 * self.acceleration_std_mps2**2
        q = np.zeros((6, 6), dtype=float)
        for axis in range(3):
            q[axis, axis] = q_pos
            q[axis + 3, axis + 3] = q_vel
            q[axis, axis + 3] = q_cross
            q[axis + 3, axis] = q_cross
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + q
        self.time_s = float(time_s)

    def update(self, measurement: np.ndarray, covariance: np.ndarray) -> None:
        h = np.zeros((3, 6), dtype=float)
        h[0, 0] = h[1, 1] = h[2, 2] = 1.0
        innovation = measurement - h @ self.state
        s = h @ self.covariance @ h.T + covariance
        k = self.covariance @ h.T @ np.linalg.pinv(s)
        self.state = self.state + k @ innovation
        self.covariance = (np.eye(6) - k @ h) @ self.covariance


def add_truth_errors(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Attach nearest/interpolated truth errors to estimate rows."""

    out = estimates.copy()
    truth_interp = _finite_truth_by_time(truth)
    estimate_times = pd.to_numeric(out["time_s"], errors="coerce").to_numpy(float)
    if truth_interp.empty:
        interp = np.full((len(out), 3), np.nan, dtype=float)
    else:
        t = truth_interp["time_s"].to_numpy(float)
        truth_xyz = truth_interp[["x_m", "y_m", "z_m"]].to_numpy(float)
        interp = np.column_stack([np.interp(estimate_times, t, truth_xyz[:, idx]) for idx in range(3)])
    est_xyz = out[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    err = est_xyz - interp
    out["truth_x_m"] = interp[:, 0]
    out["truth_y_m"] = interp[:, 1]
    out["truth_z_m"] = interp[:, 2]
    out["error_2d_m"] = np.linalg.norm(err[:, :2], axis=1)
    out["error_3d_m"] = np.linalg.norm(err, axis=1)
    return out


def _finite_truth_by_time(truth: pd.DataFrame) -> pd.DataFrame:
    columns = ["time_s", "x_m", "y_m", "z_m"]
    work = truth.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(work.to_numpy(dtype=float)).all(axis=1)
    if not finite.any():
        return work.iloc[0:0].copy()
    return work.loc[finite].groupby("time_s", as_index=False).median().sort_values("time_s")


def compute_metrics(estimates: pd.DataFrame, truth: pd.DataFrame | None) -> dict[str, Any]:
    del truth
    if estimates is None or estimates.empty or "error_3d_m" not in estimates.columns:
        return {"count": int(len(estimates)) if estimates is not None else 0}
    err3 = pd.to_numeric(estimates["error_3d_m"], errors="coerce").to_numpy(float)
    err2 = (
        pd.to_numeric(estimates["error_2d_m"], errors="coerce").to_numpy(float)
        if "error_2d_m" in estimates.columns
        else np.array([], dtype=float)
    )
    finite3 = err3[np.isfinite(err3)]
    finite2 = err2[np.isfinite(err2)]
    if finite3.size == 0:
        return {"count": 0}
    return {
        "count": int(finite3.size),
        "mean_3d_m": float(np.mean(finite3)),
        "rmse_3d_m": float(np.sqrt(np.mean(finite3**2))),
        "p95_3d_m": float(np.percentile(finite3, 95.0)),
        "max_3d_m": float(np.max(finite3)),
        "mean_2d_m": float(np.mean(finite2)) if finite2.size else None,
        "p95_2d_m": float(np.percentile(finite2, 95.0)) if finite2.size else None,
        "max_2d_m": float(np.max(finite2)) if finite2.size else None,
    }


def write_tracker_output(output: TrackerOutput, output_dir: Path) -> dict[str, str]:
    """Write estimates, selected candidates, and metrics JSON."""

    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    estimates_csv = output_dir / "mmuad_estimates.csv"
    selected_csv = output_dir / "mmuad_selected_tracklets.csv"
    metrics_json = output_dir / "mmuad_metrics.json"
    output.estimates.to_csv(estimates_csv, index=False)
    output.selected_tracklets.to_csv(selected_csv, index=False)
    metrics_json.write_text(json.dumps(output.metrics, indent=2), encoding="utf-8")
    return {
        "estimates_csv": str(estimates_csv),
        "selected_tracklets_csv": str(selected_csv),
        "metrics_json": str(metrics_json),
    }
