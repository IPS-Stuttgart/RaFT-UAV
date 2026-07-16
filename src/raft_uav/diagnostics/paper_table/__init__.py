"""Compatibility fix for robust paper-table radar interpolation.

The maintained implementation lives in the sibling ``paper_table.py`` module.
This package preserves the public import path while excluding malformed radar
anchors before interpolation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "paper_table.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._paper_table_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load paper-table implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_INTERPOLATE_SELECTED_RADAR = (
    _IMPL._interpolate_selected_radar_to_frame_times
)
_POSITION_COLUMNS = ("east_m", "north_m", "up_m")


def _finite_interpolation_anchors(selected: pd.DataFrame) -> pd.DataFrame:
    """Return anchors with finite numeric timestamps and complete 3D positions."""

    required = ("time_s", *_POSITION_COLUMNS)
    if any(column not in selected.columns for column in required):
        return selected

    anchors = selected.copy()
    for column in required:
        anchors[column] = pd.to_numeric(anchors[column], errors="coerce")
    finite = np.isfinite(
        anchors.loc[:, list(required)].to_numpy(dtype=float)
    ).all(axis=1)
    return anchors.loc[finite].copy()


def _interpolate_selected_radar_to_frame_times(
    radar: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    association_mode: str,
    max_gap_s: float | None = None,
    max_speed_mps: float | None = None,
) -> pd.DataFrame:
    """Interpolate from usable anchors without letting one bad row erase output."""

    return _ORIGINAL_INTERPOLATE_SELECTED_RADAR(
        radar,
        _finite_interpolation_anchors(selected),
        association_mode=association_mode,
        max_gap_s=max_gap_s,
        max_speed_mps=max_speed_mps,
    )


_IMPL._finite_interpolation_anchors = _finite_interpolation_anchors
_IMPL._interpolate_selected_radar_to_frame_times = (
    _interpolate_selected_radar_to_frame_times
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_interpolation_anchors"] = _finite_interpolation_anchors
globals()["_interpolate_selected_radar_to_frame_times"] = (
    _interpolate_selected_radar_to_frame_times
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
