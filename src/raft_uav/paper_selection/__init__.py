"""Compatibility fixes for paper-style radar preselection.

The maintained implementation lives in the sibling ``paper_selection.py``
module. This package preserves the public import path while validating numeric
gate parameters, excluding malformed or out-of-range class probabilities,
preserving exact integer-like track identifiers, splitting reused frame counters
into distinct continuous-track epochs, using the acquisition cadence for
timestamp-only continuity, and rejecting pooled physical-flight inputs at the
single-track boundary.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float, optional_int

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "paper_selection.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav._paper_selection_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load paper-selection implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)
_ORIGINAL_CONTINUOUS_TRACK_SEGMENTS = _LEGACY._continuous_track_segments
_ORIGINAL_RANGE_CANDIDATE_POOL = _LEGACY._range_candidate_pool
_ORIGINAL_REQUIRE_FORTEM_RANGE_M = _LEGACY.require_fortem_range_m


def _finite_nonnegative_value(value: object, *, name: str) -> float:
    """Return one finite non-Boolean scalar greater than or equal to zero."""

    parsed = optional_float(value)
    if parsed is None or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return parsed


def _finite_unit_interval_value(value: object, *, name: str) -> float:
    """Return one finite non-Boolean scalar in the closed unit interval."""

    parsed = optional_float(value)
    if parsed is None or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite real scalar in [0, 1]")
    return parsed


def _finite_catprob_value(value: object) -> float | None:
    """Return one finite real probability in [0, 1] without lossy coercion."""

    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if scalar.ndim != 0:
        return None
    item = scalar.item()
    if np.ma.is_masked(item) or isinstance(item, (bool, np.bool_)):
        return None
    if np.iscomplexobj(item):
        if not np.isfinite(item.real) or not np.isfinite(item.imag) or item.imag != 0.0:
            return None
        item = item.real
    parsed = optional_float(item)
    if parsed is None or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _finite_catprob_values(values: pd.Series) -> np.ndarray:
    """Return bounded finite class probabilities and mark malformed cells missing."""

    parsed = [_finite_catprob_value(value) for value in values.tolist()]
    return np.asarray(
        [np.nan if value is None else value for value in parsed],
        dtype=float,
    )


def _range_candidate_pool(
    candidates: pd.DataFrame,
    *,
    range_gate_m: float | None,
    require_range_m: bool,
) -> pd.DataFrame:
    """Apply a finite non-negative range gate or preserve the disabled ``None`` gate."""

    validated_range_gate_m = (
        None
        if range_gate_m is None
        else _finite_nonnegative_value(range_gate_m, name="range_gate_m")
    )
    return _ORIGINAL_RANGE_CANDIDATE_POOL(
        candidates,
        range_gate_m=validated_range_gate_m,
        require_range_m=require_range_m,
    )


def _catprob_candidate_pool(
    candidates: pd.DataFrame,
    catprob_threshold: float | None,
) -> pd.DataFrame:
    """Apply a valid class-probability gate without accepting invalid values."""

    threshold = (
        None
        if catprob_threshold is None
        else _finite_unit_interval_value(
            catprob_threshold,
            name="catprob_threshold",
        )
    )
    if threshold is None or "cat_prob_uav" not in candidates.columns:
        return candidates.copy()
    catprob = _finite_catprob_values(candidates["cat_prob_uav"])
    pool = candidates.loc[
        np.isfinite(catprob) & (catprob >= threshold)
    ].copy()
    if not pool.empty:
        pool["association_catprob_threshold"] = threshold
        pool["association_catprob_candidate_rows"] = int(len(candidates))
    return pool


def require_fortem_range_m(
    radar: pd.DataFrame,
    *,
    minimum_finite_fraction: float = 0.99,
) -> None:
    """Validate the requested finite-range fraction before checking Fortem ranges."""

    validated_fraction = _finite_unit_interval_value(
        minimum_finite_fraction,
        name="minimum_finite_fraction",
    )
    _ORIGINAL_REQUIRE_FORTEM_RANGE_M(
        radar,
        minimum_finite_fraction=validated_fraction,
    )


def _mean_catprob(frame: pd.DataFrame) -> float:
    """Return the mean of valid class probabilities for track tie-breaking."""

    if "cat_prob_uav" not in frame.columns or frame.empty:
        return 0.0
    catprob = _finite_catprob_values(frame["cat_prob_uav"])
    finite = catprob[np.isfinite(catprob)]
    return float(np.mean(finite)) if finite.size else 0.0


def _track_id_from_frame(frame: pd.DataFrame) -> int:
    """Return the first exact integer-like track ID without float round-trips."""

    if "track_id" not in frame.columns:
        return -1
    for value in frame["track_id"].tolist():
        track_id = optional_int(value)
        if track_id is not None:
            return track_id
    return -1


def _explicit_scope_ids(radar: pd.DataFrame, column: str) -> set[str]:
    """Return normalized, non-missing identifiers for one physical-scope field."""

    if radar.empty or column not in radar.columns:
        return set()
    values = radar[column].astype("string").str.strip()
    return set(values.loc[values.notna() & values.ne("")].tolist())


def _explicit_sequence_ids(radar: pd.DataFrame) -> set[str]:
    """Return normalized, non-missing sequence identifiers from one radar table."""

    return _explicit_scope_ids(radar, "sequence_id")


def _explicit_flight_ids(radar: pd.DataFrame) -> set[str]:
    """Return normalized, non-missing physical-flight identifiers from one radar table."""

    return _explicit_scope_ids(radar, "flight_id")


def _timestamp_gap_threshold(radar: pd.DataFrame) -> float:
    """Estimate timestamp continuity from the complete radar acquisition."""

    if "time_s" not in radar.columns:
        return float("inf")
    values = pd.to_numeric(
        radar["time_s"],
        errors="coerce",
    ).to_numpy(dtype=float)
    return float(_LEGACY._segment_gap_threshold(values))


def _timestamp_track_segments(
    track_rows: pd.DataFrame,
    *,
    gap_threshold: float,
) -> list[pd.DataFrame]:
    """Split one timestamp-only track using a shared acquisition threshold."""

    sort_columns = [
        column for column in ("time_s", "track_index") if column in track_rows.columns
    ]
    ordered = track_rows.sort_values(
        sort_columns,
        kind="mergesort",
    ).reset_index(drop=True)
    times = pd.to_numeric(
        ordered["time_s"],
        errors="coerce",
    ).to_numpy(dtype=float)
    split_points = np.r_[
        0,
        np.where(np.diff(times) > gap_threshold)[0] + 1,
        len(ordered),
    ]
    return [
        ordered.iloc[int(start) : int(end)].copy()
        for start, end in zip(split_points[:-1], split_points[1:])
        if int(end) > int(start)
    ]


def _continuous_track_segments(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Split one physical flight at frame restarts or acquisition-scale timestamp gaps."""

    sequence_ids = _explicit_sequence_ids(radar)
    if len(sequence_ids) > 1:
        raise ValueError(
            "paper radar track selection requires one sequence_id; "
            "split pooled radar data by sequence"
        )
    flight_ids = _explicit_flight_ids(radar)
    if len(flight_ids) > 1:
        raise ValueError(
            "paper radar track selection requires one flight_id; "
            "split pooled radar data by physical flight"
        )
    if radar.empty or "track_id" not in radar.columns:
        return []

    timestamp_gap_threshold = _timestamp_gap_threshold(radar)
    segments: list[pd.DataFrame] = []
    for _, track_rows in radar.groupby("track_id", sort=True):
        frame_index = (
            pd.to_numeric(track_rows["frame_index"], errors="coerce")
            if "frame_index" in track_rows.columns
            else None
        )
        if (
            frame_index is None
            or not bool(np.isfinite(frame_index).all())
            or "time_s" not in track_rows.columns
        ):
            if "time_s" not in track_rows.columns:
                segments.extend(_ORIGINAL_CONTINUOUS_TRACK_SEGMENTS(track_rows))
                continue
            times = pd.to_numeric(track_rows["time_s"], errors="coerce")
            if not bool(np.isfinite(times).all()):
                segments.extend(_ORIGINAL_CONTINUOUS_TRACK_SEGMENTS(track_rows))
                continue
            segments.extend(
                _timestamp_track_segments(
                    track_rows,
                    gap_threshold=timestamp_gap_threshold,
                )
            )
            continue

        times = pd.to_numeric(track_rows["time_s"], errors="coerce")
        if not bool(np.isfinite(times).all()):
            segments.extend(_ORIGINAL_CONTINUOUS_TRACK_SEGMENTS(track_rows))
            continue

        chronological_order = np.argsort(
            times.to_numpy(dtype=float),
            kind="mergesort",
        )
        chronological = track_rows.iloc[chronological_order].reset_index(drop=True)
        chronological_frames = pd.to_numeric(
            chronological["frame_index"],
            errors="coerce",
        ).to_numpy(dtype=float)
        chronological_times = pd.to_numeric(
            chronological["time_s"],
            errors="coerce",
        ).to_numpy(dtype=float)
        frame_deltas = np.diff(chronological_frames)
        time_deltas = np.diff(chronological_times)
        restart = (time_deltas > 1.0e-9) & (frame_deltas <= 1.0e-9)
        epoch_bounds = np.r_[
            0,
            np.flatnonzero(restart) + 1,
            len(chronological),
        ]
        for start, end in zip(epoch_bounds[:-1], epoch_bounds[1:]):
            epoch = chronological.iloc[int(start) : int(end)].copy()
            segments.extend(_ORIGINAL_CONTINUOUS_TRACK_SEGMENTS(epoch))
    return segments


_LEGACY._range_candidate_pool = _range_candidate_pool
_LEGACY._catprob_candidate_pool = _catprob_candidate_pool
_LEGACY.require_fortem_range_m = require_fortem_range_m
_LEGACY._mean_catprob = _mean_catprob
_LEGACY._track_id_from_frame = _track_id_from_frame
_LEGACY._timestamp_gap_threshold = _timestamp_gap_threshold
_LEGACY._timestamp_track_segments = _timestamp_track_segments
_LEGACY._continuous_track_segments = _continuous_track_segments

globals().update(
    {
        name: getattr(_LEGACY, name)
        for name in dir(_LEGACY)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_nonnegative_value"] = _finite_nonnegative_value
globals()["_finite_unit_interval_value"] = _finite_unit_interval_value
globals()["_finite_catprob_value"] = _finite_catprob_value
globals()["_finite_catprob_values"] = _finite_catprob_values
globals()["_range_candidate_pool"] = _range_candidate_pool
globals()["_catprob_candidate_pool"] = _catprob_candidate_pool
globals()["require_fortem_range_m"] = require_fortem_range_m
globals()["_mean_catprob"] = _mean_catprob
globals()["_track_id_from_frame"] = _track_id_from_frame
globals()["_explicit_scope_ids"] = _explicit_scope_ids
globals()["_explicit_sequence_ids"] = _explicit_sequence_ids
globals()["_explicit_flight_ids"] = _explicit_flight_ids
globals()["_timestamp_gap_threshold"] = _timestamp_gap_threshold
globals()["_timestamp_track_segments"] = _timestamp_track_segments
globals()["_continuous_track_segments"] = _continuous_track_segments

__doc__ = _LEGACY.__doc__
__all__ = [
    name for name in dir(_LEGACY) if not (name.startswith("__") and name.endswith("__"))
]
