"""Compatibility fix for empty Track 5 submission-ensemble weight grids.

The maintained implementation lives in the sibling
``track5_submission_ensemble_grid.py`` module. This package preserves the public
import path while rejecting an empty weight grid before pandas attempts to sort
an empty, column-less result frame.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_submission_ensemble_grid.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_submission_ensemble_grid_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load submission-ensemble grid implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_EVALUATE_SUBMISSION_ENSEMBLE_WEIGHT_GRID = (
    _IMPL.evaluate_submission_ensemble_weight_grid
)


def evaluate_submission_ensemble_weight_grid(
    submission_inputs: Iterable[Any],
    *,
    truth: pd.DataFrame,
    weight_grid: Iterable[tuple[float, ...]],
    class_policies: Iterable[str] = ("weighted-vote",),
    timestamp_tolerance_s: float = 1.0e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, ...], str]:
    """Score a non-empty weight grid with the maintained implementation."""

    weight_rows = tuple(weight_grid)
    if not weight_rows:
        raise ValueError("weight grid produced no rows")
    return _ORIGINAL_EVALUATE_SUBMISSION_ENSEMBLE_WEIGHT_GRID(
        submission_inputs,
        truth=truth,
        weight_grid=weight_rows,
        class_policies=class_policies,
        timestamp_tolerance_s=timestamp_tolerance_s,
    )


_IMPL.evaluate_submission_ensemble_weight_grid = evaluate_submission_ensemble_weight_grid

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["evaluate_submission_ensemble_weight_grid"] = (
    evaluate_submission_ensemble_weight_grid
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
