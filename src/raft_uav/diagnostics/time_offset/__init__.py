from __future__ import annotations

from contextvars import ContextVar
import importlib.util
from pathlib import Path

import numpy as _np
import pandas as _pd

from raft_uav.numeric import optional_float as _optional_float
from raft_uav.numeric import optional_int as _optional_int

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
_original_resolve_dimensions = _legacy.resolve_dimensions
_original_run_time_offset_diagnostic = _legacy.run_time_offset_diagnostic
_ACTIVE_SWEEP_TAU_BOUNDS: ContextVar[tuple[float, float] | None] = ContextVar(
    "raft_uav_time_offset_sweep_tau_bounds",
    default=None,
)

_POSITION_COLUMNS = ("east_m", "north_m", "up_m")


def _finite_real_control(value: object, *, name: str) -> float:
    """Return one finite real scalar without Boolean or array coercion."""

    normalized = _optional_float(value)
    if normalized is None:
        raise ValueError(f"{name} must be a finite real scalar")
    return normalized


def _positive_real_control(value: object, *, name: str) -> float:
    """Return one finite positive real scalar."""

    normalized = _finite_real_control(value, name=name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return normalized


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
    threshold = _finite_real_control(threshold, name="threshold")
    return _original_catprob_candidate_pool(candidates, threshold)


def highest_catprob_candidate(candidates):
    return _original_highest_catprob_candidate(_finite_position_candidates(candidates))


def nearest_candidate_to_truth(candidates, truth_position):
    return _original_nearest_candidate_to_truth(
        _finite_position_candidates(candidates),
        truth_position,
    )


def resolve_dimensions(source: str, dimensions: object) -> int:
    """Resolve ``auto`` or validate an explicit 2D/3D selection."""

    if isinstance(dimensions, str) and dimensions == "auto":
        return _original_resolve_dimensions(source, dimensions)
    normalized = _optional_int(dimensions)
    if normalized not in {2, 3}:
        raise ValueError("dimensions must be 'auto', 2, or 3")
    return normalized


def _radar_group_sort_key(values: _pd.Series) -> _pd.Series:
    """Normalize numeric frame keys before sorting serialized radar tables."""

    if values.name == "time_s":
        return values.map(_optional_float)
    if values.name == "frame_index":
        return _pd.Series(
            [_optional_int(value) for value in values],
            index=values.index,
            dtype=object,
        )
    return values


def radar_frame_groups(radar: _pd.DataFrame) -> list[_pd.DataFrame]:
    """Group every radar row while preserving each usable physical frame ID."""

    if radar.empty:
        return []
    if "time_s" not in radar.columns and "frame_index" not in radar.columns:
        raise KeyError("radar must contain time_s or frame_index")
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    ordered = radar.sort_values(
        sort_columns,
        key=_radar_group_sort_key,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    group_positions: dict[tuple[str, object], list[int]] = {}
    for position, row in ordered.iterrows():
        frame_index = (
            _optional_int(row["frame_index"])
            if "frame_index" in ordered.columns
            else None
        )
        if frame_index is not None:
            key: tuple[str, object] = ("frame_index", frame_index)
        elif "time_s" in ordered.columns:
            time_s = _optional_float(row["time_s"])
            key = (
                ("time_s", time_s)
                if time_s is not None
                else ("row", int(position))
            )
        else:
            key = ("row", int(position))
        group_positions.setdefault(key, []).append(int(position))

    return [
        ordered.iloc[positions].copy()
        for positions in group_positions.values()
    ]


def _longest_track_id(radar: _pd.DataFrame) -> int | None:
    """Return the most frequent exact integer track identifier."""

    if "track_id" not in radar.columns or radar.empty:
        return None
    track_ids = _pd.Series(
        [_optional_int(value) for value in radar["track_id"]],
        index=radar.index,
        dtype=object,
    ).dropna()
    if track_ids.empty:
        return None
    counts = track_ids.value_counts()
    return int(counts.idxmax())


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
    radar_catprob_threshold: float | None = 0.4,
    max_truth_time_delta_s: float = 2.0,
    objective: str = "p95",
    output_dir: Path = Path("outputs/time-offset"),
    write_plot: bool = True,
):
    """Run an offset sweep after validating numeric controls before data access."""

    normalized_tau_min_s = _finite_real_control(tau_min_s, name="tau_min_s")
    normalized_tau_max_s = _finite_real_control(tau_max_s, name="tau_max_s")
    normalized_tau_step_s = _positive_real_control(tau_step_s, name="tau_step_s")
    normalized_dimensions = resolve_dimensions(source, dimensions)
    normalized_max_delta_s = _positive_real_control(
        max_truth_time_delta_s,
        name="max_truth_time_delta_s",
    )
    normalized_catprob_threshold = radar_catprob_threshold
    if source == "radar" and radar_catprob_threshold is not None:
        normalized_catprob_threshold = _finite_real_control(
            radar_catprob_threshold,
            name="radar_catprob_threshold",
        )

    token = _ACTIVE_SWEEP_TAU_BOUNDS.set(
        (normalized_tau_min_s, normalized_tau_max_s)
    )
    try:
        return _original_run_time_offset_diagnostic(
            dataset_root=dataset_root,
            flight_name=flight_name,
            source=source,
            tau_min_s=normalized_tau_min_s,
            tau_max_s=normalized_tau_max_s,
            tau_step_s=normalized_tau_step_s,
            dimensions=normalized_dimensions,
            radar_selection=radar_selection,
            radar_catprob_threshold=normalized_catprob_threshold,
            max_truth_time_delta_s=normalized_max_delta_s,
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
_legacy._longest_track_id = _longest_track_id
_legacy.resolve_dimensions = resolve_dimensions
_legacy._inside_truth_window = _inside_truth_window
_legacy.run_time_offset_diagnostic = run_time_offset_diagnostic

for _name, _value in vars(_legacy).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
globals()["catprob_candidate_pool"] = catprob_candidate_pool
globals()["highest_catprob_candidate"] = highest_catprob_candidate
globals()["nearest_candidate_to_truth"] = nearest_candidate_to_truth
globals()["radar_frame_groups"] = radar_frame_groups
globals()["_longest_track_id"] = _longest_track_id
globals()["resolve_dimensions"] = resolve_dimensions
globals()["run_time_offset_diagnostic"] = run_time_offset_diagnostic
__all__ = sorted(_name for _name in globals() if not _name.startswith("_"))
