from __future__ import annotations

from contextvars import ContextVar
import importlib.util
from pathlib import Path

import numpy as _np
import pandas as _pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "time_offset.py"
_SPEC = importlib.util.spec_from_file_location("_raft_uav_time_offset_legacy", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_legacy)
_original_catprob_candidate_pool = _legacy.catprob_candidate_pool
_original_highest_catprob_candidate = _legacy.highest_catprob_candidate
_original_nearest_candidate_to_truth = _legacy.nearest_candidate_to_truth
_original_inside_truth_window = _legacy._inside_truth_window
_original_run_time_offset_diagnostic = _legacy.run_time_offset_diagnostic
_ACTIVE_SWEEP_TAU_BOUNDS: ContextVar[tuple[float, float] | None] = ContextVar(
    "raft_uav_time_offset_sweep_tau_bounds",
    default=None,
)

_POSITION_COLUMNS = ("east_m", "north_m", "up_m")


def _finite_position_candidates(candidates):
    if candidates.empty:
        return candidates

    numeric_positions = candidates.loc[:, _POSITION_COLUMNS].apply(
        _pd.to_numeric,
        errors="coerce",
    )
    finite = _np.isfinite(numeric_positions.to_numpy(dtype=float)).all(axis=1)
    cleaned = candidates.loc[finite].copy()
    for column in _POSITION_COLUMNS:
        cleaned[column] = numeric_positions.loc[finite, column].to_numpy(dtype=float)
    return cleaned


def catprob_candidate_pool(candidates, threshold):
    if threshold is None:
        return candidates
    return _original_catprob_candidate_pool(candidates, threshold)


def highest_catprob_candidate(candidates):
    return _original_highest_catprob_candidate(_finite_position_candidates(candidates))


def nearest_candidate_to_truth(candidates, truth_position):
    return _original_nearest_candidate_to_truth(
        _finite_position_candidates(candidates),
        truth_position,
    )


def radar_frame_groups(radar: _pd.DataFrame) -> list[_pd.DataFrame]:
    """Group every radar row even when ``frame_index`` is partly populated."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    if not sort_columns:
        raise KeyError("radar must contain time_s or frame_index")
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    frame_index_complete = (
        "frame_index" in ordered.columns and ordered["frame_index"].notna().all()
    )
    if frame_index_complete:
        group_column = "frame_index"
    elif "time_s" in ordered.columns:
        group_column = "time_s"
    else:
        raise KeyError("radar must contain time_s when frame_index is incomplete")
    return [
        group.copy()
        for _, group in ordered.groupby(group_column, sort=True, dropna=False)
    ]


def _inside_truth_window(frame: _pd.DataFrame, truth: _pd.DataFrame) -> _pd.DataFrame:
    """Keep rows that can overlap truth for at least one active sweep offset."""

    bounds = _ACTIVE_SWEEP_TAU_BOUNDS.get()
    if bounds is None:
        return _original_inside_truth_window(frame, truth)
    if frame.empty or "time_s" not in frame.columns:
        return frame

    tau_min_s, tau_max_s = bounds
    start = float(truth["time_s"].min()) - tau_max_s
    end = float(truth["time_s"].max()) - tau_min_s
    return frame.loc[(frame["time_s"] >= start) & (frame["time_s"] <= end)].copy()


def run_time_offset_diagnostic(
    *,
    dataset_root: Path,
    flight_name: str,
    source: str,
    tau_min_s: float,
    tau_max_s: float,
    tau_step_s: float,
    dimensions: str = "auto",
    radar_selection: str = "oracle-nearest-truth",
    radar_catprob_threshold: float = 0.4,
    max_truth_time_delta_s: float = 2.0,
    objective: str = "p95",
    output_dir: Path = Path("outputs/time-offset"),
    write_plot: bool = True,
):
    """Run an offset sweep without discarding rows that shift into truth support."""

    token = _ACTIVE_SWEEP_TAU_BOUNDS.set((float(tau_min_s), float(tau_max_s)))
    try:
        return _original_run_time_offset_diagnostic(
            dataset_root=dataset_root,
            flight_name=flight_name,
            source=source,
            tau_min_s=tau_min_s,
            tau_max_s=tau_max_s,
            tau_step_s=tau_step_s,
            dimensions=dimensions,
            radar_selection=radar_selection,
            radar_catprob_threshold=radar_catprob_threshold,
            max_truth_time_delta_s=max_truth_time_delta_s,
            objective=objective,
            output_dir=output_dir,
            write_plot=write_plot,
        )
    finally:
        _ACTIVE_SWEEP_TAU_BOUNDS.reset(token)


_legacy.catprob_candidate_pool = catprob_candidate_pool
_legacy.highest_catprob_candidate = highest_catprob_candidate
_legacy.nearest_candidate_to_truth = nearest_candidate_to_truth
_legacy.radar_frame_groups = radar_frame_groups
_legacy._inside_truth_window = _inside_truth_window
_legacy.run_time_offset_diagnostic = run_time_offset_diagnostic

for _name, _value in vars(_legacy).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
globals()["catprob_candidate_pool"] = catprob_candidate_pool
globals()["highest_catprob_candidate"] = highest_catprob_candidate
globals()["nearest_candidate_to_truth"] = nearest_candidate_to_truth
globals()["radar_frame_groups"] = radar_frame_groups
globals()["run_time_offset_diagnostic"] = run_time_offset_diagnostic
__all__ = sorted(_name for _name in globals() if not _name.startswith("_"))
