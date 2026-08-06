"""Compatibility fixes for comprehensive radar-frame diagnostics.

The maintained implementation lives in the sibling
``comprehensive_improvements.py`` module. This package preserves the public
import path while keeping radar rows whose optional ``frame_index`` is missing,
separating reused frame counters by timestamp, and rejecting unusable truth
matches.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import TypeAlias

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "comprehensive_improvements.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.research._comprehensive_improvements_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise ImportError(
        f"cannot load comprehensive-improvements implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_FrameKey: TypeAlias = tuple[str, int | float, float | None]


def _diagnostic_frame_key(
    row: pd.Series,
    *,
    fallback_position: int,
) -> _FrameKey:
    """Return a collision-safe frame key while preserving public key fields.

    The first two tuple entries retain the established ``(key_type, key)``
    contract used in diagnostic tables. A hidden timestamp component prevents
    a reused finite frame counter from collapsing physically distinct frames.
    """

    time_s = _IMPL._finite_float(row.get("time_s"))
    rounded_time_s = None if time_s is None else round(float(time_s), 9)
    if "frame_index" in row.index:
        frame_index = _IMPL._finite_float(row.get("frame_index"))
        if frame_index is not None:
            return "frame_index", int(frame_index), rounded_time_s
    if rounded_time_s is not None:
        return "time_s", rounded_time_s, None
    return "row_position", int(fallback_position), None


def _frame_key(frame: pd.DataFrame) -> _FrameKey:
    """Return the collision-safe identity for one already grouped frame."""

    if frame.empty:
        return "row_position", 0, None
    return _diagnostic_frame_key(frame.iloc[0], fallback_position=0)


def _row_key(row: pd.Series) -> _FrameKey:
    """Return the same collision-safe identity for one selected radar row."""

    fallback_position = int(row.name) if isinstance(row.name, (int, np.integer)) else 0
    return _diagnostic_frame_key(row, fallback_position=fallback_position)


def _sort_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric sort series or an all-missing fallback."""

    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Group radar rows with row-wise fallback from frame index to timestamp.

    Pandas drops null group keys by default. Selecting ``frame_index`` merely
    because the column exists therefore removed otherwise valid rows whenever
    that optional column was only partially populated. Serialized missing
    markers had the opposite failure mode and collapsed unrelated timestamps
    into one synthetic frame. A row-local key avoids both outcomes, while its
    hidden timestamp component also separates reused finite frame counters.
    """

    if radar is None or radar.empty:
        return []
    source = pd.DataFrame(radar).copy()
    positions = np.arange(len(source), dtype=int)
    track_column = next(
        (column for column in ("track_id", "track_index") if column in source.columns),
        None,
    )
    if track_column is None:
        track_numeric = pd.Series(np.nan, index=source.index, dtype=float)
        track_text = pd.Series("", index=source.index, dtype="string")
    else:
        track_numeric = pd.to_numeric(source[track_column], errors="coerce")
        track_text = source[track_column].astype("string").fillna("")

    order = pd.DataFrame(
        {
            "position": positions,
            "time_s": _sort_series(source, "time_s").to_numpy(),
            "frame_index": _sort_series(source, "frame_index").to_numpy(),
            "track_numeric": track_numeric.to_numpy(),
            "track_text": track_text.to_numpy(),
        }
    ).sort_values(
        ["time_s", "frame_index", "track_numeric", "track_text", "position"],
        kind="mergesort",
        na_position="last",
    )

    positions_by_key: dict[_FrameKey, list[int]] = {}
    for position in order["position"].astype(int):
        key = _diagnostic_frame_key(
            source.iloc[position],
            fallback_position=int(position),
        )
        positions_by_key.setdefault(key, []).append(int(position))
    return [
        source.iloc[group_positions].copy()
        for group_positions in positions_by_key.values()
    ]


def _nearest_truth_position(
    truth: pd.DataFrame,
    *,
    time_s: float,
    max_delta_s: float,
) -> tuple[np.ndarray | None, float | None]:
    """Return the nearest finite truth sample for a finite query timestamp."""

    query_time_s = _IMPL._finite_float(time_s)
    max_delta = _IMPL._finite_float(max_delta_s)
    required = ["time_s", "east_m", "north_m", "up_m"]
    if (
        query_time_s is None
        or max_delta is None
        or max_delta < 0.0
        or truth is None
        or truth.empty
        or not all(column in truth.columns for column in required)
    ):
        return None, None

    times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(dtype=float)
    positions = truth[["east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    usable = np.isfinite(times) & np.isfinite(positions).all(axis=1)
    if not bool(usable.any()):
        return None, None

    usable_indices = np.flatnonzero(usable)
    local_index = int(
        _IMPL._nearest_time_indices(
            times[usable],
            np.array([float(query_time_s)], dtype=float),
        )[0]
    )
    index = int(usable_indices[local_index])
    delta_s = float(abs(times[index] - float(query_time_s)))
    if delta_s > float(max_delta):
        return None, delta_s
    return positions[index], delta_s


_IMPL._diagnostic_frame_key = _diagnostic_frame_key
_IMPL._frame_key = _frame_key
_IMPL._row_key = _row_key
_IMPL._radar_frame_groups = _radar_frame_groups
_IMPL._nearest_truth_position = _nearest_truth_position

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_FrameKey"] = _FrameKey
globals()["_diagnostic_frame_key"] = _diagnostic_frame_key
globals()["_frame_key"] = _frame_key
globals()["_row_key"] = _row_key
globals()["_sort_series"] = _sort_series
globals()["_radar_frame_groups"] = _radar_frame_groups
globals()["_nearest_truth_position"] = _nearest_truth_position

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
