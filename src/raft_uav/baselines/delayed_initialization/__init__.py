"""Compatibility wrapper for usable-row radar initialization windows.

The maintained implementation lives in the sibling
``delayed_initialization.py`` module. This package preserves the public import
path while ensuring that malformed early radar rows cannot anchor the delayed
initialization window and exclude later usable detections.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "delayed_initialization.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._delayed_initialization_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load delayed-initialization implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _first_radar_window(radar: pd.DataFrame, *, window_s: float) -> pd.DataFrame:
    """Return the first window anchored at the earliest usable 3D radar row."""

    position_columns = tuple(_IMPL._POSITION_COLUMNS)
    if (
        radar.empty
        or "time_s" not in radar.columns
        or not set(position_columns).issubset(radar.columns)
    ):
        return radar.iloc[0:0].copy()

    work = radar.copy()
    times = work["time_s"].map(_IMPL._optional_float)
    positions = work.loc[:, position_columns].apply(
        lambda column: column.map(_IMPL._optional_float)
    )
    usable = times.notna() & positions.notna().all(axis=1)
    work = work.loc[usable].copy()
    if work.empty:
        return work

    work["time_s"] = times.loc[usable].astype(float)
    ordered = work.sort_values("time_s", kind="mergesort").reset_index(drop=True)
    start = float(ordered["time_s"].iloc[0])
    return ordered.loc[ordered["time_s"] <= start + window_s].copy()


_IMPL._first_radar_window = _first_radar_window

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_first_radar_window"] = _first_radar_window

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
