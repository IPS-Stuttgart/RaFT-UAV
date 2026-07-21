"""Compatibility fixes for Track 5 RTS ensemble grid inputs and scoring.

The implementation lives in the sibling ``track5_rts_ensemble_grid.py`` file.
This wrapper preserves opaque estimate sequence IDs and rejects malformed
truth-matching tolerances before they can silently widen or empty train-side
scoring.
"""

from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_rts_ensemble_grid.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_rts_ensemble_grid_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 RTS ensemble grid implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


class _PandasEstimateCsvProxy:
    """Delegate pandas operations while preserving estimate identifier columns."""

    def __init__(self, pandas_module: Any) -> None:
        self._pandas = pandas_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pandas, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if args or kwargs:
            return self._pandas.read_csv(path, *args, **kwargs)
        return read_estimate_csv(Path(path))


_IMPL.pd = _PandasEstimateCsvProxy(pd)
_ORIGINAL_RUN_GRID_SEARCH = _IMPL.run_track5_rts_ensemble_grid_search


def _nonnegative_finite_scalar(value: object, *, name: str) -> float:
    """Return a validated non-negative finite scalar control."""

    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be a non-negative finite scalar")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative finite scalar") from exc
    if array.ndim != 0 or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a non-negative finite scalar")
    try:
        normalized = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative finite scalar") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a non-negative finite scalar")
    return normalized


@wraps(_ORIGINAL_RUN_GRID_SEARCH)
def run_track5_rts_ensemble_grid_search(
    *args: object,
    score_time_tolerance_s: object = 1.0e-6,
    **kwargs: object,
):
    """Run the grid search with a validated truth-time scoring tolerance."""

    tolerance = _nonnegative_finite_scalar(
        score_time_tolerance_s,
        name="score_time_tolerance_s",
    )
    return _ORIGINAL_RUN_GRID_SEARCH(
        *args,
        score_time_tolerance_s=tolerance,
        **kwargs,
    )


_IMPL.run_track5_rts_ensemble_grid_search = run_track5_rts_ensemble_grid_search

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["run_track5_rts_ensemble_grid_search"] = run_track5_rts_ensemble_grid_search

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
