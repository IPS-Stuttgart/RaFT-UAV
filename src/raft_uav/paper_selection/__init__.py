"""Compatibility fixes for paper-style radar preselection.

The maintained implementation lives in the sibling ``paper_selection.py``
module. This package preserves the public import path while excluding malformed
or out-of-range class probabilities, preserving exact integer-like track
identifiers, splitting reused frame counters into distinct continuous-track
epochs, and using the acquisition cadence for timestamp-only continuity.
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


def _catprob_candidate_pool(
    candidates: pd.DataFrame,
    catprob_threshold: float | None,
) -> pd.DataFrame:
    """Apply the class-probability gate without accepting invalid values."""

    if catprob_threshold is None or "cat_prob_uav" not in candidates.columns:
        return candidates.copy()
    catprob = _finite_catprob_values(candidates["cat_prob_uav"])
    pool = candidates.loc[
        np.isfinite(catprob) & (catprob >= float(catprob_threshold))
    ].copy()
    if not pool.empty:
        pool["association_catprob_threshold"] = float(catprob_threshold)
        pool["association_catprob_candidate_rows"] = int(len(candidates))
    return pool


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
    """Split tracks at frame restarts or acquisition-scale timestamp gaps."""

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


_LEGACY._catprob_candidate_pool = _catprob_candidate_pool
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
globals()["_finite_catprob_value"] = _finite_catprob_value
globals()["_finite_catprob_values"] = _finite_catprob_values
globals()["_catprob_candidate_pool"] = _catprob_candidate_pool
globals()["_mean_catprob"] = _mean_catprob
globals()["_track_id_from_frame"] = _track_id_from_frame
globals()["_timestamp_gap_threshold"] = _timestamp_gap_threshold
globals()["_timestamp_track_segments"] = _timestamp_track_segments
globals()["_continuous_track_segments"] = _continuous_track_segments

__doc__ = _LEGACY.__doc__
__all__ = [
    name for name in dir(_LEGACY) if not (name.startswith("__") and name.endswith("__"))
]
