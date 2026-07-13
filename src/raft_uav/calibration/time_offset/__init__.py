"""Compatibility wrapper correcting pooled time-offset error statistics.

The maintained implementation lives in the sibling ``time_offset.py`` module.
This package preserves the public import path while computing pooled population
standard deviations from the aggregated first and second moments.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "time_offset.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._time_offset_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load time-offset calibration implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_AGGREGATE_ERROR_FRAMES = _IMPL._aggregate_error_frames


def _pooled_population_std(mean: float, rmse: float) -> float:
    """Recover population standard deviation from pooled first/second moments."""

    mean_value = float(mean)
    rmse_value = float(rmse)
    if not np.isfinite(mean_value) or not np.isfinite(rmse_value):
        return float("nan")
    variance = rmse_value**2 - mean_value**2
    scale = max(rmse_value**2, mean_value**2, 1.0)
    tolerance = 32.0 * np.finfo(float).eps * scale
    if variance < -tolerance:
        return float("nan")
    return float(np.sqrt(max(variance, 0.0)))


def _aggregate_error_frames(
    offset: float,
    frames: list[pd.DataFrame],
    frame_count: int,
) -> dict[str, float]:
    """Aggregate summaries without discarding between-flight variation."""

    row = _LEGACY_AGGREGATE_ERROR_FRAMES(offset, frames, frame_count)
    if not frames:
        return row
    for dims in ("3d", "2d"):
        row[f"std_{dims}_error_m"] = _pooled_population_std(
            row[f"mean_{dims}_error_m"],
            row[f"rmse_{dims}_error_m"],
        )
    return row


_IMPL._aggregate_error_frames = _aggregate_error_frames

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep corrected helpers importable for focused regressions.
globals()["_pooled_population_std"] = _pooled_population_std
globals()["_aggregate_error_frames"] = _aggregate_error_frames

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
