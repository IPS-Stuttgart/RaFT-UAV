"""Shared paper-style Fortem radar preselection utilities.

The AERPAW RF + Fortem radar paper baseline is very sensitive to the exact
radar stream that enters the Kalman filter.  Several RaFT-UAV entry points used
slightly different implementations of the same idea: Fortem range gating,
optional UAV class-probability gating, and largest-continuous-track retention.

This module is intentionally small and dataframe-oriented so strict paper
diagnostics, paper-compatible online fusion, and paper-table diagnostics can all
share the same preselector instead of drifting independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

PAPER_RADAR_TRACK_SELECTION_ORDERS = (
    "raw-track-then-range",
    "range-then-largest-track",
    "range-catprob-then-largest-track",
)
DEFAULT_PAPER_RADAR_TRACK_SELECTION_ORDER = "raw-track-then-range"


@dataclass(frozen=True)
class PaperRadarTrackStages:
    """Radar target-stream stages used by strict paper-parity diagnostics."""

    raw_target: pd.DataFrame
    range_gated: pd.DataFrame
    preselected: pd.DataFrame


def paper_radar_track_stages(
    radar: pd.DataFrame,
    *,
    range_gate_m: float | None = 800.0,
    catprob_threshold: float | None = None,
    require_range_m: bool = True,
    radar_track_selection_order: str = DEFAULT_PAPER_RADAR_TRACK_SELECTION_ORDER,
) -> PaperRadarTrackStages:
    """Return raw-target, range-gated, and update-ready radar stages.

    The reference paper mentions both an 800 m radar range gate and largest
    continuous Fortem-track retention.  The text does not fully disambiguate
    whether the largest track is chosen before or after range/class filtering,
    so the ordering is explicit and recorded in row metadata.
    """

    _validate_radar_track_selection_order(radar_track_selection_order)
    if require_range_m:
        require_fortem_range_m(radar)

    if radar_track_selection_order == "raw-track-then-range":
        raw_target = select_paper_strict_raw_radar_track(radar)
        range_gated = range_gated_radar_candidates(
            raw_target,
            range_gate_m=range_gate_m,
            catprob_threshold=catprob_threshold,
            require_range_m=require_range_m,
        )
        preselected = _annotate_strict_update_radar(
            range_gated,
            raw_radar=radar,
            raw_target_radar=raw_target,
            range_gate_m=range_gate_m,
            catprob_threshold=catprob_threshold,
            radar_track_selection_order=radar_track_selection_order,
            association_action="range_gated_raw_track_anchor",
        )
        return PaperRadarTrackStages(raw_target, range_gated, preselected)

    if radar_track_selection_order == "range-then-largest-track":
        range_pool = range_gated_radar_candidates(
            radar,
            range_gate_m=range_gate_m,
            catprob_threshold=None,
            require_range_m=require_range_m,
        )
        raw_target = _annotate_strict_raw_target_track(
            _largest_continuous_track_segment(range_pool),
            raw_radar=radar,
            candidate_rows=len(range_pool),
            radar_track_selection_order=radar_track_selection_order,
            association_mode="paper-strict-range-then-largest-track",
            association_action="range_then_largest_track_anchor",
        )
        range_gated = _catprob_candidate_pool(raw_target, catprob_threshold)
        preselected = _annotate_strict_update_radar(
            range_gated,
            raw_radar=radar,
            raw_target_radar=raw_target,
            range_gate_m=range_gate_m,
            catprob_threshold=catprob_threshold,
            radar_track_selection_order=radar_track_selection_order,
            association_action="range_then_largest_track_anchor",
        )
        return PaperRadarTrackStages(raw_target, range_gated, preselected)

    range_catprob_pool = range_gated_radar_candidates(
        radar,
        range_gate_m=range_gate_m,
        catprob_threshold=catprob_threshold,
        require_range_m=require_range_m,
    )
    raw_target = _annotate_strict_raw_target_track(
        _largest_continuous_track_segment(range_catprob_pool),
        raw_radar=radar,
        candidate_rows=len(range_catprob_pool),
        radar_track_selection_order=radar_track_selection_order,
        association_mode="paper-strict-range-catprob-then-largest-track",
        association_action="range_catprob_then_largest_track_anchor",
    )
    preselected = _annotate_strict_update_radar(
        raw_target,
        raw_radar=radar,
        raw_target_radar=raw_target,
        range_gate_m=range_gate_m,
        catprob_threshold=catprob_threshold,
        radar_track_selection_order=radar_track_selection_order,
        association_action="range_catprob_then_largest_track_anchor",
    )
    return PaperRadarTrackStages(raw_target, raw_target.copy(), preselected)


def select_paper_update_radar_track(
    radar: pd.DataFrame,
    *,
    range_gate_m: float | None = 800.0,
    catprob_threshold: float | None = None,
    require_range_m: bool = True,
    radar_track_selection_order: str = DEFAULT_PAPER_RADAR_TRACK_SELECTION_ORDER,
) -> pd.DataFrame:
    """Return the update-ready radar target stream for one paper-style order."""

    return paper_radar_track_stages(
        radar,
        range_gate_m=range_gate_m,
        catprob_threshold=catprob_threshold,
        require_range_m=require_range_m,
        radar_track_selection_order=radar_track_selection_order,
    ).preselected


def select_paper_compatible_radar_track(
    radar: pd.DataFrame,
    *,
    range_gate_m: float | None,
    catprob_threshold: float | None,
) -> pd.DataFrame:
    """Return anchors for the online ``paper-compatible`` association mode.

    This preserves the historical online behavior: apply the hard range gate,
    apply the optional hard class-probability gate without fallback, then retain
    the largest continuous Fortem track segment from that candidate pool.
    """

    range_pool = range_gated_radar_candidates(
        radar,
        range_gate_m=range_gate_m,
        catprob_threshold=None,
        require_range_m=range_gate_m is not None,
    )
    catprob_pool = _catprob_candidate_pool(range_pool, catprob_threshold)
    selected = _largest_continuous_track_segment(catprob_pool)
    if selected.empty:
        return selected

    selected = selected.copy()
    selected["association_mode"] = "paper-compatible"
    selected["association_action"] = "paper_compatible_largest_continuous_track_anchor"
    selected["association_segment_count"] = 1
    selected["association_segment_track_id"] = _track_id_from_frame(selected)
    selected["association_segment_frames"] = int(len(selected))
    selected["association_segment_start_time_s"] = float(selected["time_s"].iloc[0])
    selected["association_segment_end_time_s"] = float(selected["time_s"].iloc[-1])
    selected["association_preselector_raw_rows"] = int(len(radar))
    selected["association_preselector_range_gated_rows"] = int(len(range_pool))
    selected["association_preselector_track_id"] = int(selected["association_segment_track_id"].iloc[0])
    selected["association_preselector_track_rows"] = int(len(selected))
    selected["association_preselector_catprob_rows"] = int(len(catprob_pool))
    if range_gate_m is not None:
        selected["association_range_gate_m"] = float(range_gate_m)
    if catprob_threshold is not None:
        selected["association_catprob_threshold"] = float(catprob_threshold)
    return _sort_radar_rows(selected).reset_index(drop=True)


def range_gated_radar_candidates(
    radar: pd.DataFrame,
    *,
    range_gate_m: float | None = 800.0,
    catprob_threshold: float | None = None,
    require_range_m: bool = True,
) -> pd.DataFrame:
    """Return radar rows after Fortem range gating and optional hard catProb."""

    if require_range_m:
        require_fortem_range_m(radar)
    pool = _range_candidate_pool(
        radar,
        range_gate_m=range_gate_m,
        require_range_m=require_range_m,
    )
    pool = _catprob_candidate_pool(pool, catprob_threshold)
    return _sort_radar_rows(pool).reset_index(drop=True)


def select_paper_strict_raw_radar_track(radar: pd.DataFrame) -> pd.DataFrame:
    """Select the largest continuous Fortem track before validation gates."""

    selected = _largest_continuous_track_segment(radar)
    if selected.empty:
        return selected
    return _annotate_strict_raw_target_track(
        selected,
        raw_radar=radar,
        candidate_rows=len(radar),
        radar_track_selection_order=DEFAULT_PAPER_RADAR_TRACK_SELECTION_ORDER,
        association_mode="paper-strict-raw-largest-continuous-track",
        association_action="raw_largest_continuous_track_anchor",
    )


def require_fortem_range_m(
    radar: pd.DataFrame,
    *,
    minimum_finite_fraction: float = 0.99,
) -> None:
    """Fail if strict range gating would have to fall back to ENU distance."""

    if radar.empty:
        return
    if "range_m" not in radar.columns:
        raise ValueError(
            "paper range gating requires Fortem range_m; pass an explicit "
            "non-parity option only when ENU-norm fallback is intended"
        )
    ranges = pd.to_numeric(radar["range_m"], errors="coerce").to_numpy(dtype=float)
    finite_fraction = float(np.mean(np.isfinite(ranges))) if ranges.size else 0.0
    if finite_fraction < float(minimum_finite_fraction):
        raise ValueError(
            "paper range gating requires finite range_m for at least "
            f"{float(minimum_finite_fraction):.1%} of rows; observed "
            f"{finite_fraction:.3f}"
        )


def _annotate_strict_raw_target_track(
    selected: pd.DataFrame,
    *,
    raw_radar: pd.DataFrame,
    candidate_rows: int,
    radar_track_selection_order: str,
    association_mode: str,
    association_action: str,
) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    out = selected.copy()
    out["association_mode"] = association_mode
    out["association_action"] = association_action
    out["association_track_selection_order"] = radar_track_selection_order
    out["association_preselector_raw_rows"] = int(len(raw_radar))
    out["association_preselector_candidate_rows"] = int(candidate_rows)
    out["association_raw_target_track_rows"] = int(len(out))
    return _sort_radar_rows(out).reset_index(drop=True)


def _annotate_strict_update_radar(
    selected: pd.DataFrame,
    *,
    raw_radar: pd.DataFrame,
    raw_target_radar: pd.DataFrame,
    range_gate_m: float | None,
    catprob_threshold: float | None,
    radar_track_selection_order: str,
    association_action: str,
) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    out = selected.copy()
    out["association_mode"] = "paper-strict-largest-continuous-track"
    out["association_action"] = association_action
    out["association_track_selection_order"] = radar_track_selection_order
    if range_gate_m is not None:
        out["association_range_gate_m"] = float(range_gate_m)
    out["association_preselector_raw_rows"] = int(len(raw_radar))
    out["association_raw_target_track_rows"] = int(len(raw_target_radar))
    out["association_preselector_range_gated_rows"] = int(len(out))
    out["association_segment_frames"] = int(len(out))
    if catprob_threshold is not None:
        out["association_catprob_threshold"] = float(catprob_threshold)
    return _sort_radar_rows(out).reset_index(drop=True)


def _largest_continuous_track_segment(radar: pd.DataFrame) -> pd.DataFrame:
    if radar.empty or "track_id" not in radar.columns:
        return radar.iloc[0:0].copy()
    segments = _continuous_track_segments(radar)
    if not segments:
        return radar.iloc[0:0].copy()
    selected_segment = max(
        segments,
        key=lambda segment: (
            int(len(segment)),
            float(segment["time_s"].iloc[-1] - segment["time_s"].iloc[0]),
            _mean_catprob(segment),
            -float(segment["time_s"].iloc[0]),
            -_track_id_from_frame(segment),
        ),
    )
    return selected_segment.copy()


def _continuous_track_segments(radar: pd.DataFrame) -> list[pd.DataFrame]:
    if radar.empty or "track_id" not in radar.columns:
        return []
    segments: list[pd.DataFrame] = []
    for _, track_rows in radar.groupby("track_id", sort=True):
        sort_columns = [
            column
            for column in ("frame_index", "time_s", "track_index")
            if column in track_rows.columns
        ]
        ordered = track_rows.sort_values(sort_columns).reset_index(drop=True)
        frame_values = (
            pd.to_numeric(ordered["frame_index"], errors="coerce").to_numpy(dtype=float)
            if "frame_index" in ordered.columns
            else ordered["time_s"].to_numpy(dtype=float)
        )
        split_points = np.r_[
            0,
            np.where(np.diff(frame_values) > _segment_gap_threshold(frame_values))[0] + 1,
            len(ordered),
        ]
        for start, end in zip(split_points[:-1], split_points[1:]):
            segment = ordered.iloc[int(start) : int(end)].copy()
            if not segment.empty:
                segments.append(segment)
    return segments


def _range_candidate_pool(
    candidates: pd.DataFrame,
    *,
    range_gate_m: float | None,
    require_range_m: bool,
) -> pd.DataFrame:
    if candidates.empty or range_gate_m is None:
        return candidates.copy()
    ranges, range_source = _candidate_ranges_m(
        candidates,
        require_range_m=require_range_m,
    )
    pool = candidates.loc[np.isfinite(ranges) & (ranges <= float(range_gate_m))].copy()
    if not pool.empty:
        pool["association_range_gate_m"] = float(range_gate_m)
        pool["association_range_source"] = range_source
        pool["association_range_fortem_required"] = bool(require_range_m)
    return pool


def _candidate_ranges_m(
    candidates: pd.DataFrame,
    *,
    require_range_m: bool,
) -> tuple[np.ndarray, str]:
    if "range_m" in candidates.columns:
        ranges = pd.to_numeric(candidates["range_m"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(ranges)
        if finite.any():
            return ranges, "range_m"
        if require_range_m:
            raise ValueError("paper range gating found no finite Fortem range_m values")
    elif require_range_m:
        raise ValueError(
            "paper range gating requires Fortem range_m; falling back to ENU "
            "norm changes the 800 m gate"
        )
    positions = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    return np.linalg.norm(positions, axis=1), "enu_norm_fallback"


def _catprob_candidate_pool(
    candidates: pd.DataFrame,
    catprob_threshold: float | None,
) -> pd.DataFrame:
    if catprob_threshold is None or "cat_prob_uav" not in candidates.columns:
        return candidates.copy()
    catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce")
    pool = candidates.loc[catprob >= float(catprob_threshold)].copy()
    if not pool.empty:
        pool["association_catprob_threshold"] = float(catprob_threshold)
        pool["association_catprob_candidate_rows"] = int(len(candidates))
    return pool


def _sort_radar_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in frame.columns
    ]
    if not sort_columns:
        return frame.copy()
    return frame.sort_values(sort_columns, kind="mergesort")


def _track_id_from_frame(frame: pd.DataFrame) -> int:
    values = pd.to_numeric(frame["track_id"], errors="coerce").dropna()
    if values.empty:
        return -1
    return int(values.iloc[0])


def _mean_catprob(frame: pd.DataFrame) -> float:
    if "cat_prob_uav" not in frame.columns or frame.empty:
        return 0.0
    catprob = pd.to_numeric(frame["cat_prob_uav"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(catprob).any():
        return 0.0
    return float(np.nanmean(catprob))


def _segment_gap_threshold(frame_values: np.ndarray) -> float:
    values = np.sort(np.asarray(frame_values, dtype=float).reshape(-1))
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("inf")
    diffs = np.diff(values)
    positive = diffs[diffs > 1.0e-9]
    if positive.size == 0:
        return float("inf")
    if _integer_like(values):
        return 1.5
    return 1.5 * float(np.median(positive))


def _integer_like(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return bool(finite.size and np.allclose(finite, np.round(finite)))


def _validate_radar_track_selection_order(order: str) -> None:
    if order not in PAPER_RADAR_TRACK_SELECTION_ORDERS:
        raise ValueError(
            f"radar_track_selection_order must be one of "
            f"{PAPER_RADAR_TRACK_SELECTION_ORDERS}, got {order!r}"
        )
