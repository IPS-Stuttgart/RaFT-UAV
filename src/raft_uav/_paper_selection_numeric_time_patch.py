"""Normalize numeric-like paper-selection ordering without mutating row payloads."""

from __future__ import annotations

from importlib import import_module
from typing import Sequence

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float


_paper_selection = import_module("raft_uav.paper_selection")
_PATCH_MARKER = "_raft_uav_paper_numeric_time_patch_applied"


def _numeric_series(values: pd.Series) -> pd.Series:
    """Return finite real scalar values while preserving the source index."""

    return pd.Series(
        [optional_float(value) for value in values],
        index=values.index,
        dtype=float,
    )


def _sort_with_numeric_keys(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Sort numeric-like keys numerically while retaining original columns."""

    if frame.empty:
        return frame.copy()

    out = frame.copy()
    sort_columns: list[str] = []
    temporary_columns: list[str] = []
    for position, column in enumerate(columns):
        if column not in out.columns:
            continue

        numeric = _numeric_series(out[column])
        nonmissing = ~out[column].isna()
        use_numeric = column in {"time_s", "frame_index"} or bool(
            numeric.loc[nonmissing].notna().all()
        )
        if not use_numeric:
            sort_columns.append(column)
            continue

        key = f"__raft_uav_numeric_sort_{position}"
        while key in out.columns:
            key += "_"
        out[key] = numeric
        sort_columns.append(key)
        temporary_columns.append(key)

    if not sort_columns:
        return out
    ordered = out.sort_values(
        sort_columns,
        kind="mergesort",
        na_position="last",
    )
    return ordered.drop(columns=temporary_columns)


def _continuous_track_segments(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Split tracks after stable numeric ordering of frame and time keys."""

    if radar.empty or "track_id" not in radar.columns:
        return []

    segments: list[pd.DataFrame] = []
    for _, track_rows in radar.groupby("track_id", sort=True):
        frame_index = (
            _numeric_series(track_rows["frame_index"])
            if "frame_index" in track_rows.columns
            else None
        )
        use_frame_index = frame_index is not None and bool(
            np.isfinite(frame_index).all()
        )
        sort_candidates = (
            ("frame_index", "time_s", "track_index")
            if use_frame_index
            else ("time_s", "track_index")
        )
        ordered = _sort_with_numeric_keys(track_rows, sort_candidates).reset_index(
            drop=True
        )
        frame_column = "frame_index" if use_frame_index else "time_s"
        frame_values = _numeric_series(ordered[frame_column]).to_numpy(dtype=float)
        split_points = np.r_[
            0,
            np.where(
                np.diff(frame_values)
                > _paper_selection._segment_gap_threshold(frame_values)
            )[0]
            + 1,
            len(ordered),
        ]
        for start, end in zip(split_points[:-1], split_points[1:]):
            segment = ordered.iloc[int(start) : int(end)].copy()
            if not segment.empty:
                segments.append(segment)
    return segments


def _largest_continuous_track_segment(radar: pd.DataFrame) -> pd.DataFrame:
    """Select the largest segment using numeric timestamp tie breakers."""

    if radar.empty or "track_id" not in radar.columns:
        return radar.iloc[0:0].copy()
    segments = _continuous_track_segments(radar)
    if not segments:
        return radar.iloc[0:0].copy()

    def segment_key(segment: pd.DataFrame) -> tuple[int, float, float, float, int]:
        times = _numeric_series(segment["time_s"]).to_numpy(dtype=float)
        finite_times = times[np.isfinite(times)]
        if finite_times.size:
            start_time = float(finite_times[0])
            duration = float(finite_times[-1] - finite_times[0])
        else:
            start_time = float("inf")
            duration = float("-inf")
        return (
            int(len(segment)),
            duration,
            _paper_selection._mean_catprob(segment),
            -start_time,
            -_paper_selection._track_id_from_frame(segment),
        )

    return max(segments, key=segment_key).copy()


def _sort_radar_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort public paper-selection output by normalized numeric keys."""

    return _sort_with_numeric_keys(
        frame,
        ("time_s", "frame_index", "track_id", "track_index"),
    )


def install() -> None:
    """Install numeric paper-selection ordering once per interpreter."""

    if getattr(_paper_selection, _PATCH_MARKER, False):
        return
    _paper_selection._continuous_track_segments = _continuous_track_segments
    _paper_selection._largest_continuous_track_segment = (
        _largest_continuous_track_segment
    )
    _paper_selection._sort_radar_rows = _sort_radar_rows
    setattr(_paper_selection, _PATCH_MARKER, True)
