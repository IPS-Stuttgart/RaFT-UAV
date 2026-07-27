"""Preserve physical radar frame boundaries with partial frame indices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_INSTALLED = False


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Group indexed frames exactly and fall back per row to finite timestamps."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    times = pd.to_numeric(ordered["time_s"], errors="coerce")
    if "frame_index" in ordered.columns:
        frame_indices = pd.to_numeric(ordered["frame_index"], errors="coerce")
    else:
        frame_indices = pd.Series(np.nan, index=ordered.index, dtype=float)
    group_keys = pd.Series(
        [
            ("frame_index_time", float(frame_index), float(time_s))
            if np.isfinite(frame_index) and np.isfinite(time_s)
            else ("frame_index", float(frame_index))
            if np.isfinite(frame_index)
            else ("time_s", float(time_s))
            if np.isfinite(time_s)
            else None
            for frame_index, time_s in zip(frame_indices, times, strict=True)
        ],
        index=ordered.index,
        dtype=object,
    )
    usable = group_keys.notna()
    ordered = ordered.loc[usable]
    group_keys = group_keys.loc[usable]
    return [group.copy() for _, group in ordered.groupby(group_keys, sort=False)]


def install() -> None:
    """Install collision-safe frame grouping into radar association."""

    global _INSTALLED
    if _INSTALLED:
        return
    from raft_uav.baselines import radar_association

    radar_association._radar_frame_groups = _radar_frame_groups
    implementation: Any = getattr(radar_association, "_IMPL", None)
    if implementation is not None:
        implementation._radar_frame_groups = _radar_frame_groups
    _INSTALLED = True
