"""Compatibility fixes for radar-track segment fingerprint diagnostics.

The maintained implementation lives in the sibling ``radar_fingerprint.py``
module. This package preserves the public import path while keeping frame-index
continuity chronological and separating track segments when counters reset.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "radar_fingerprint.py"
_LEGACY_NAME = f"{__name__.rsplit('.', 1)[0]}._radar_fingerprint_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load radar fingerprint implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

for _name in dir(_LEGACY):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_LEGACY, _name)


def _continuous_track_segments(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Split tracks chronologically, including at frame-counter resets."""

    if radar.empty or "track_id" not in radar.columns:
        return []
    segments: list[pd.DataFrame] = []
    for _, track_rows in radar.groupby("track_id", sort=True):
        continuity_key = _LEGACY._track_continuity_key(track_rows)
        order_columns = (
            ["time_s"]
            if continuity_key == "time_s"
            else ["time_s", continuity_key]
        )
        ordered = track_rows.sort_values(
            order_columns,
            kind="mergesort",
        ).reset_index(drop=True)
        values = pd.to_numeric(
            ordered[continuity_key],
            errors="coerce",
        ).to_numpy(dtype=float)
        deltas = np.diff(values)
        threshold = _LEGACY._segment_gap_threshold(values)
        discontinuities = (deltas < 0.0) | (deltas > threshold)
        split_points = np.r_[
            0,
            np.where(discontinuities)[0] + 1,
            len(ordered),
        ]
        for start, end in zip(split_points[:-1], split_points[1:]):
            segment = ordered.iloc[int(start) : int(end)].copy()
            if not segment.empty:
                segments.append(segment)
    return segments


_LEGACY._continuous_track_segments = _continuous_track_segments
