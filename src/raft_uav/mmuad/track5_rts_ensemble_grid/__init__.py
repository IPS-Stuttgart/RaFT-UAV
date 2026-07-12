"""Compatibility fixes for Track 5 RTS ensemble grid search.

The implementation lives in the sibling ``track5_rts_ensemble_grid.py`` file.
This wrapper keeps the public import path while preserving opaque sequence IDs
and making one-pass parameter-grid iterables reusable across the full Cartesian
product.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput

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


def run_track5_rts_ensemble_grid_search(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    measurement_sigma_grid: Iterable[float] = _IMPL.DEFAULT_MEASUREMENT_SIGMA_GRID,
    process_accel_grid: Iterable[float] = _IMPL.DEFAULT_PROCESS_ACCEL_GRID,
    spread_variance_scale_grid: Iterable[float] = _IMPL.DEFAULT_SPREAD_VARIANCE_SCALE_GRID,
    initial_position_std_m: float = 100.0,
    initial_velocity_std_mps: float = 25.0,
    max_nearest_time_delta_s: float | None = None,
    score_time_tolerance_s: float = 1.0e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate every grid combination when callers supply generators.

    The implementation validates the process and spread grids inside outer
    loops. Tuples are re-iterable, but generators are exhausted after the first
    outer iteration. Materializing each documented ``Iterable`` once preserves
    the complete Cartesian product.
    """

    measurement_sigma_values = tuple(measurement_sigma_grid)
    process_accel_values = tuple(process_accel_grid)
    spread_variance_scale_values = tuple(spread_variance_scale_grid)
    return _ORIGINAL_RUN_GRID_SEARCH(
        estimate_inputs,
        template=template,
        truth=truth,
        measurement_sigma_grid=measurement_sigma_values,
        process_accel_grid=process_accel_values,
        spread_variance_scale_grid=spread_variance_scale_values,
        initial_position_std_m=initial_position_std_m,
        initial_velocity_std_mps=initial_velocity_std_mps,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        score_time_tolerance_s=score_time_tolerance_s,
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
