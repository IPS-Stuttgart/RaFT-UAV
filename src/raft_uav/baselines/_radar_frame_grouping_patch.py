"""Preserve physical radar frame boundaries with partial frame indices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_INSTALLED = False


def _radar_frame_group_keys(radar: pd.DataFrame) -> pd.Series:
    """Return collision-safe per-row frame keys with timestamp fallbacks."""

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

    keys: list[tuple[Any, ...] | None] = []
    for frame_index, timestamp, time_s in zip(
        frame_indices, timestamps, times, strict=True
    ):
        time_key: tuple[Any, ...] | None = None
        if not pd.isna(timestamp):
            time_key = ("timestamp", timestamp)
        elif np.isfinite(time_s):
            time_key = ("time_s", float(time_s))

        if np.isfinite(frame_index):
            if time_key is None:
                key = ("frame_index", float(frame_index))
            else:
                key = ("frame_index", float(frame_index), *time_key)
        else:
            key = time_key
        keys.append(key)
    return pd.Series(keys, index=radar.index, dtype=object)


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Group indexed frames exactly and fall back per row to finite timestamps."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    ordered = (
        radar.sort_values(sort_columns).reset_index(drop=True)
        if sort_columns
        else radar.reset_index(drop=True)
    )
    group_keys = _radar_frame_group_keys(ordered)
    usable = group_keys.notna()
    ordered = ordered.loc[usable]
    group_keys = group_keys.loc[usable]
    return [group.copy() for _, group in ordered.groupby(group_keys, sort=False)]


def _catprob_best_per_frame_rows(
    radar: pd.DataFrame, catprob_threshold: float
) -> pd.DataFrame:
    """Select one candidate per physical frame under mixed index availability."""

    from raft_uav.io import aerpaw

    candidates = aerpaw._catprob_threshold_rows(radar, catprob_threshold)
    if candidates.empty:
        return candidates

    group_keys = [
        key if key is not None else ("__missing_frame__",)
        for key in _radar_frame_group_keys(candidates).tolist()
    ]
    key_column = "__raft_uav_frame_group_key__"
    while key_column in candidates.columns:
        key_column = f"_{key_column}"
    ranked = aerpaw._catprob_ranked_rows(
        candidates.assign(**{key_column: group_keys})
    )

    keep_positions: list[int] = []
    seen_frames: set[tuple[Any, ...]] = set()
    for position, frame_key in enumerate(ranked[key_column].tolist()):
        if frame_key in seen_frames:
            continue
        seen_frames.add(frame_key)
        keep_positions.append(position)
    selected = aerpaw._dataframe_from_ranked_records(ranked, keep_positions)
    return selected.drop(columns=[key_column]).sort_index()


def install() -> None:
    """Install collision-safe grouping into association and legacy selection."""

    global _INSTALLED
    if _INSTALLED:
        return
    from raft_uav.baselines import radar_association
    from raft_uav.io import aerpaw

    radar_association._radar_frame_groups = _radar_frame_groups
    implementation: Any = getattr(radar_association, "_IMPL", None)
    if implementation is not None:
        implementation._radar_frame_groups = _radar_frame_groups
    aerpaw._catprob_best_per_frame_rows = _catprob_best_per_frame_rows
    _INSTALLED = True
