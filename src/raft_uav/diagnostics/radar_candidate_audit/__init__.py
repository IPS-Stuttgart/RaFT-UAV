"""Compatibility fixes for radar candidate-audit physical-frame grouping.

The maintained implementation lives in the sibling ``radar_candidate_audit.py``
module. This package preserves the public import path while keeping partially
indexed and reused radar frames distinct.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "radar_candidate_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._radar_candidate_audit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load radar candidate audit from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _finite_frame_value(value: Any) -> float | None:
    """Return a finite numeric frame value or ``None`` when unavailable."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _frame_key(frame: pd.DataFrame) -> pd.Series:
    """Return collision-safe physical-frame keys for candidate-audit rows.

    A finite frame index is qualified by time so counters that restart later in
    a flight remain separate. Missing indices fall back to time. When exactly
    one indexed frame exists at that time, missing-index candidates join it.
    Rows without usable frame metadata remain distinct rather than collapsing
    into one synthetic frame.
    """

    if "frame_index" in frame.columns:
        frame_indices = frame["frame_index"].tolist()
    else:
        frame_indices = [None] * len(frame)

    time_column = "audit_time_s" if "audit_time_s" in frame.columns else "time_s"
    if time_column in frame.columns:
        times = frame[time_column].tolist()
    else:
        times = [None] * len(frame)

    rows: list[tuple[float | None, float | None]] = []
    indexed_frames_by_time: dict[float, set[float]] = {}
    for frame_index, time_s in zip(frame_indices, times, strict=True):
        numeric_frame = _finite_frame_value(frame_index)
        numeric_time = _finite_frame_value(time_s)
        rounded_time = round(numeric_time, 9) if numeric_time is not None else None
        rows.append((numeric_frame, rounded_time))
        if numeric_frame is not None and rounded_time is not None:
            indexed_frames_by_time.setdefault(rounded_time, set()).add(numeric_frame)

    keys: list[tuple[object, ...]] = []
    for row_position, (frame_index, time_s) in enumerate(rows):
        if frame_index is not None and time_s is not None:
            key = ("frame_index_time_s", frame_index, time_s)
        elif frame_index is not None:
            key = ("frame_index", frame_index)
        elif time_s is not None:
            matching_frames = indexed_frames_by_time.get(time_s, set())
            if len(matching_frames) == 1:
                key = ("frame_index_time_s", next(iter(matching_frames)), time_s)
            else:
                key = ("time_s", time_s)
        else:
            key = ("__missing_frame__", row_position)
        keys.append(key)
    return pd.Series(keys, index=frame.index, dtype=object)


_IMPL._frame_key = _frame_key

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_frame_value"] = _finite_frame_value
globals()["_frame_key"] = _frame_key

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
