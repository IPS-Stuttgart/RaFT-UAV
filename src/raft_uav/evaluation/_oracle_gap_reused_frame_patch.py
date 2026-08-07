"""Keep reused oracle-gap frame counters physically separate."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from importlib import import_module

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float


_oracle_gap = import_module("raft_uav.evaluation.oracle_gap_decomposition")
_ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP = _oracle_gap.decompose_radar_oracle_gap
_ORIGINAL_FRAME_KEY = _oracle_gap._frame_key
_ORIGINAL_ROW_KEY = _oracle_gap._row_key
_PATCH_MARKER = "_reused_frame_index_patch_applied"
_REUSED_FRAME_INDICES: ContextVar[
    dict[str | None, frozenset[float]] | None
] = ContextVar("raft_uav_oracle_gap_reused_frame_indices", default=None)


def _sequence_keys(frame: pd.DataFrame) -> pd.Series:
    """Return the sequence normalization used by the existing scope patch."""

    if "sequence_id" not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype=object)
    return (
        pd.Series(frame["sequence_id"], index=frame.index, dtype="string")
        .str.strip()
        .fillna("")
    )


def _reused_frame_indices(
    radar: pd.DataFrame,
) -> dict[str | None, frozenset[float]]:
    """Find frame counters observed at multiple finite timestamps per sequence."""

    if radar.empty or "frame_index" not in radar.columns or "time_s" not in radar.columns:
        return {}
    frame_indices = pd.to_numeric(radar["frame_index"], errors="coerce")
    times = pd.to_numeric(radar["time_s"], errors="coerce")
    usable = np.isfinite(frame_indices.to_numpy(dtype=float)) & np.isfinite(
        times.to_numpy(dtype=float)
    )
    if not bool(usable.any()):
        return {}

    working = pd.DataFrame(
        {
            "sequence_key": _sequence_keys(radar),
            "frame_index": frame_indices,
            "time_key": times.round(9),
        },
        index=radar.index,
    ).loc[usable]
    reused: dict[str | None, set[float]] = {}
    for (sequence_key, frame_index), group in working.groupby(
        ["sequence_key", "frame_index"],
        dropna=False,
        sort=False,
    ):
        if group["time_key"].nunique(dropna=True) > 1:
            normalized_sequence = None if pd.isna(sequence_key) else str(sequence_key)
            reused.setdefault(normalized_sequence, set()).add(float(frame_index))
    return {key: frozenset(values) for key, values in reused.items()}


def _frame_scope(frame: pd.DataFrame) -> str | None:
    if frame.empty or "sequence_id" not in frame.columns:
        return None
    keys = _sequence_keys(frame)
    return str(keys.iloc[0]) if not keys.empty else None


def _row_scope(row: pd.Series) -> str | None:
    if "sequence_id" not in row.index:
        return None
    return str(
        pd.Series([row.get("sequence_id")], dtype="string").str.strip().fillna("").iloc[0]
    )


def _indices_for_scope(scope: str | None) -> frozenset[float]:
    reused = _REUSED_FRAME_INDICES.get()
    if not reused:
        return frozenset()
    if scope in reused:
        return reused[scope]
    if scope is None and len(reused) == 1:
        return next(iter(reused.values()))
    return frozenset()


def _is_reused(frame_index: float, *, scope: str | None) -> bool:
    return float(frame_index) in _indices_for_scope(scope)


def _qualified_frame_key(frame_index: float, time_s: float) -> str:
    """Return one stable key for a reused counter at one physical timestamp."""

    return f"{int(frame_index)}@{round(float(time_s), 9):.9f}"


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Qualify only reused finite frame indices by their physical timestamp."""

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
    scopes = _sequence_keys(ordered)

    group_keys: list[tuple[object, ...] | None] = []
    for frame_index, time_s, scope in zip(frame_indices, times, scopes, strict=True):
        normalized_scope = None if pd.isna(scope) else str(scope)
        if np.isfinite(frame_index):
            if np.isfinite(time_s) and _is_reused(
                float(frame_index),
                scope=normalized_scope,
            ):
                group_keys.append(
                    (
                        "frame_index_time_s",
                        float(frame_index),
                        round(float(time_s), 9),
                    )
                )
            else:
                group_keys.append(("frame_index", float(frame_index)))
        elif np.isfinite(time_s):
            group_keys.append(("time_s", round(float(time_s), 9)))
        else:
            group_keys.append(None)

    keys = pd.Series(group_keys, index=ordered.index, dtype=object)
    usable = keys.notna()
    ordered = ordered.loc[usable]
    keys = keys.loc[usable]
    return [group.copy() for _, group in ordered.groupby(keys, sort=False)]


def _frame_key(frame: pd.DataFrame) -> tuple[str, int | float | str]:
    """Expose a timestamp-qualified key only when a frame counter is reused."""

    if "frame_index" in frame.columns:
        values = pd.to_numeric(frame["frame_index"], errors="coerce").dropna()
        if not values.empty:
            frame_index = float(values.iloc[0])
            time_s = optional_float(pd.to_numeric(frame["time_s"], errors="coerce").median())
            if time_s is not None and _is_reused(
                frame_index,
                scope=_frame_scope(frame),
            ):
                return (
                    "frame_index_time_s",
                    _qualified_frame_key(frame_index, time_s),
                )
    return _ORIGINAL_FRAME_KEY(frame)


def _row_key(row: pd.Series) -> tuple[str, int | float | str]:
    """Use the same physical-frame key for selected-radar rows."""

    frame_index = optional_float(row.get("frame_index"))
    time_s = optional_float(row.get("time_s"))
    if (
        frame_index is not None
        and time_s is not None
        and _is_reused(frame_index, scope=_row_scope(row))
    ):
        return (
            "frame_index_time_s",
            _qualified_frame_key(frame_index, time_s),
        )
    return _ORIGINAL_ROW_KEY(row)


@wraps(_ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP)
def decompose_radar_oracle_gap(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    selected_radar: pd.DataFrame | None = None,
    estimates: pd.DataFrame | None = None,
    config: object | None = None,
) -> pd.DataFrame:
    """Evaluate with a call-local map of reused physical frame counters."""

    radar_rows = pd.DataFrame(radar).copy()
    token = _REUSED_FRAME_INDICES.set(_reused_frame_indices(radar_rows))
    try:
        return _ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP(
            radar=radar_rows,
            truth=truth,
            selected_radar=selected_radar,
            estimates=estimates,
            config=config,
        )
    finally:
        _REUSED_FRAME_INDICES.reset(token)


def install() -> None:
    """Install the reused-frame fix on public and maintained implementation paths."""

    if getattr(_oracle_gap, _PATCH_MARKER, False):
        return
    _oracle_gap.decompose_radar_oracle_gap = decompose_radar_oracle_gap
    _oracle_gap._radar_frame_groups = _radar_frame_groups
    _oracle_gap._frame_key = _frame_key
    _oracle_gap._row_key = _row_key
    implementation = getattr(_oracle_gap, "_IMPL", None)
    if implementation is not None:
        implementation.decompose_radar_oracle_gap = decompose_radar_oracle_gap
        implementation._radar_frame_groups = _radar_frame_groups
        implementation._frame_key = _frame_key
        implementation._row_key = _row_key
    setattr(_oracle_gap, _PATCH_MARKER, True)


install()
