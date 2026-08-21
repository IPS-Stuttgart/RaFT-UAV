"""Preserve physical radar frame boundaries with partial frame indices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_INSTALLED = False
_MISSING_SEQUENCE_KEYS = frozenset({"nan", "none", "<na>", "nat"})
_SCOPE_COLUMNS = ("sequence_id", "flight_id")


def _normalized_scope_column(radar: pd.DataFrame, column: str) -> pd.Series:
    """Return normalized scope IDs without converting missing values to text."""

    if column not in radar.columns:
        return pd.Series(None, index=radar.index, dtype=object)
    keys = pd.Series(
        radar[column],
        index=radar.index,
        dtype="string",
    ).str.strip()
    missing = keys.isna() | keys.eq("") | keys.str.lower().isin(
        _MISSING_SEQUENCE_KEYS
    )
    normalized = keys.astype(object)
    normalized.loc[missing] = None
    return normalized


def _sequence_keys(radar: pd.DataFrame) -> pd.Series:
    """Return normalized sequence IDs for backward-compatible callers."""

    return _normalized_scope_column(radar, "sequence_id")


def _scope_keys(radar: pd.DataFrame) -> pd.Series:
    """Return joint keys for every available flight-boundary alias."""

    scope_columns = [column for column in _SCOPE_COLUMNS if column in radar.columns]
    if not scope_columns:
        return pd.Series([()] * len(radar), index=radar.index, dtype=object)

    normalized_columns = [
        _normalized_scope_column(radar, column).tolist()
        for column in scope_columns
    ]
    return pd.Series(
        list(zip(*normalized_columns, strict=True)),
        index=radar.index,
        dtype=object,
    )


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Group physical frames by flight scope, index, and finite timestamp."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    scope_keys = _scope_keys(ordered)
    times = pd.to_numeric(ordered["time_s"], errors="coerce")
    if "frame_index" in ordered.columns:
        frame_indices = pd.to_numeric(ordered["frame_index"], errors="coerce")
    else:
        frame_indices = pd.Series(np.nan, index=ordered.index, dtype=float)
    group_keys = pd.Series(
        [
            (scope_key, "frame_index_time", float(frame_index), float(time_s))
            if np.isfinite(frame_index) and np.isfinite(time_s)
            else (scope_key, "frame_index", float(frame_index))
            if np.isfinite(frame_index)
            else (scope_key, "time_s", float(time_s))
            if np.isfinite(time_s)
            else None
            for scope_key, frame_index, time_s in zip(
                scope_keys,
                frame_indices,
                times,
                strict=True,
            )
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
