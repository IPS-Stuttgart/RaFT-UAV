"""Collision-safe physical-frame grouping for radar cat-probability selection."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

_aerpaw = import_module("raft_uav.io.aerpaw")


def _finite_frame_value(value: Any) -> float | None:
    """Return a finite numeric frame value or ``None`` when unavailable."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _catprob_frame_group_keys(radar: pd.DataFrame) -> list[tuple[Any, ...]]:
    """Build collision-safe per-row physical-frame keys.

    Valid frame indices remain distinct even when their timestamps collide. Rows
    without an index fall back to timestamp/time. When a timestamp contains
    exactly one indexed frame, its missing-index rows are treated as candidates
    from that same physical frame for backward compatibility. Rows without any
    usable frame metadata remain distinct because grouping them would silently
    discard unrelated detections.
    """

    if "frame_index" in radar.columns:
        frame_indices = pd.to_numeric(radar["frame_index"], errors="coerce")
    else:
        frame_indices = pd.Series(np.nan, index=radar.index, dtype=float)
    if "timestamp" in radar.columns:
        timestamps = pd.to_datetime(radar["timestamp"], errors="coerce")
    else:
        timestamps = pd.Series(pd.NaT, index=radar.index, dtype="datetime64[ns]")
    if "time_s" in radar.columns:
        times = pd.to_numeric(radar["time_s"], errors="coerce")
    else:
        times = pd.Series(np.nan, index=radar.index, dtype=float)

    rows: list[tuple[float | None, tuple[Any, ...] | None]] = []
    indexed_frames_by_time: dict[tuple[Any, ...], set[float]] = {}
    for frame_index, timestamp, time_s in zip(
        frame_indices, timestamps, times, strict=True
    ):
        if not pd.isna(timestamp):
            time_key: tuple[Any, ...] | None = ("timestamp", pd.Timestamp(timestamp))
        else:
            numeric_time = _finite_frame_value(time_s)
            time_key = ("time_s", numeric_time) if numeric_time is not None else None

        numeric_frame = _finite_frame_value(frame_index)
        rows.append((numeric_frame, time_key))
        if numeric_frame is not None and time_key is not None:
            indexed_frames_by_time.setdefault(time_key, set()).add(numeric_frame)

    keys: list[tuple[Any, ...]] = []
    for row_position, (frame_index, time_key) in enumerate(rows):
        if frame_index is not None and time_key is not None:
            key = ("frame_index", frame_index, *time_key)
        elif frame_index is not None:
            key = ("frame_index", frame_index)
        elif time_key is not None:
            matching_frames = indexed_frames_by_time.get(time_key, set())
            if len(matching_frames) == 1:
                key = ("frame_index", next(iter(matching_frames)), *time_key)
            else:
                key = time_key
        else:
            key = ("__missing_frame__", row_position)
        keys.append(key)
    return keys


def catprob_best_per_frame_rows(
    radar: pd.DataFrame, catprob_threshold: float
) -> pd.DataFrame:
    """Select the highest-probability candidate in every physical radar frame."""

    candidates = _aerpaw._catprob_threshold_rows(radar, catprob_threshold)
    if candidates.empty:
        return candidates

    key_column = "__raft_uav_frame_group_key__"
    while key_column in candidates.columns:
        key_column = f"_{key_column}"
    ranked = _aerpaw._catprob_ranked_rows(
        candidates.assign(**{key_column: _catprob_frame_group_keys(candidates)})
    )

    keep_positions: list[int] = []
    seen_frames: set[tuple[Any, ...]] = set()
    for position, frame_key in enumerate(ranked[key_column].tolist()):
        if frame_key in seen_frames:
            continue
        seen_frames.add(frame_key)
        keep_positions.append(position)
    selected = _aerpaw._dataframe_from_ranked_records(ranked, keep_positions)
    return selected.drop(columns=[key_column]).sort_index()
