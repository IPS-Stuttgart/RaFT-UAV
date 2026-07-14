"""Compatibility wrapper for robust paper-style radar segment selection.

The maintained implementation lives in the sibling ``paper_selection.py``
module. This package preserves the public import path while ensuring that
partially populated frame indices do not collapse disconnected radar rows into
one continuous track segment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "paper_selection.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav._paper_selection_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load paper selection implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _continuous_track_segments(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Split tracks without trusting an incomplete ``frame_index`` column."""

    if radar.empty or "track_id" not in radar.columns:
        return []
    segments: list[pd.DataFrame] = []
    for _, track_rows in radar.groupby("track_id", sort=True):
        frame_index_values = (
            pd.to_numeric(track_rows["frame_index"], errors="coerce").to_numpy(
                dtype=float
            )
            if "frame_index" in track_rows.columns
            else None
        )
        frame_index_complete = (
            frame_index_values is not None
            and bool(np.isfinite(frame_index_values).all())
        )
        sort_preference = (
            ("frame_index", "time_s", "track_index")
            if frame_index_complete
            else ("time_s", "frame_index", "track_index")
        )
        sort_columns = [
            column for column in sort_preference if column in track_rows.columns
        ]
        ordered = track_rows.sort_values(sort_columns).reset_index(drop=True)
        frame_values = (
            pd.to_numeric(ordered["frame_index"], errors="coerce").to_numpy(
                dtype=float
            )
            if frame_index_complete
            else pd.to_numeric(ordered["time_s"], errors="coerce").to_numpy(
                dtype=float
            )
        )
        split_points = np.r_[
            0,
            np.where(
                np.diff(frame_values)
                > _IMPL._segment_gap_threshold(frame_values)
            )[0]
            + 1,
            len(ordered),
        ]
        for start, end in zip(split_points[:-1], split_points[1:]):
            segment = ordered.iloc[int(start) : int(end)].copy()
            if not segment.empty:
                segments.append(segment)
    return segments


_IMPL._continuous_track_segments = _continuous_track_segments

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_continuous_track_segments"] = _continuous_track_segments

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
