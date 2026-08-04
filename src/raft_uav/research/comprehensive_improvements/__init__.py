"""Compatibility fixes for comprehensive radar-frame diagnostics.

The maintained implementation lives in the sibling
``comprehensive_improvements.py`` module. This package preserves the public
import path while keeping radar rows whose optional ``frame_index`` is missing
and sorting numeric-like chronology keys numerically.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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


def _diagnostic_frame_key(
    row: pd.Series,
    *,
    fallback_position: int,
) -> tuple[str, int | float]:
    """Return a row-local frame key without dropping missing frame indices."""

    if "frame_index" in row.index:
        frame_index = _IMPL._finite_float(row.get("frame_index"))
        if frame_index is not None:
            return "frame_index", int(frame_index)
    time_s = _IMPL._finite_float(row.get("time_s"))
    if time_s is not None:
        return "time_s", round(float(time_s), 9)
    return "row_position", int(fallback_position)


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
    into one synthetic frame. A row-local key avoids both outcomes.
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

    positions_by_key: dict[tuple[str, int | float], list[int]] = {}
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


_IMPL._diagnostic_frame_key = _diagnostic_frame_key
_IMPL._radar_frame_groups = _radar_frame_groups

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_diagnostic_frame_key"] = _diagnostic_frame_key
globals()["_sort_series"] = _sort_series
globals()["_radar_frame_groups"] = _radar_frame_groups

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
