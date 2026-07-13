"""Compatibility fix for reusable Track 5 submission-grid iterables.

The maintained implementation lives in the sibling
``track5_submission_ensemble_grid.py`` module. This package preserves the public
import path while materializing one-shot weight and policy iterables before the
implementation reuses them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

from raft_uav.mmuad.track5_submission_ensemble import SubmissionInput

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_submission_ensemble_grid.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_submission_ensemble_grid_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load submission-grid implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_EVALUATE = _IMPL.evaluate_submission_ensemble_weight_grid
_ORIGINAL_WRITE = _IMPL.write_submission_ensemble_weight_grid_outputs


def _materialize_weight_grid(
    weight_grid: Iterable[tuple[float, ...]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(weight) for weight in weights) for weights in weight_grid)


def evaluate_submission_ensemble_weight_grid(
    submission_inputs: Iterable[SubmissionInput],
    *,
    truth: pd.DataFrame,
    weight_grid: Iterable[tuple[float, ...]],
    class_policies: Iterable[str] = ("weighted-vote",),
    timestamp_tolerance_s: float = 1.0e-6,
):
    """Score every requested weight and class-policy combination exactly once."""

    return _ORIGINAL_EVALUATE(
        tuple(submission_inputs),
        truth=truth,
        weight_grid=_materialize_weight_grid(weight_grid),
        class_policies=tuple(class_policies),
        timestamp_tolerance_s=timestamp_tolerance_s,
    )


def write_submission_ensemble_weight_grid_outputs(
    *,
    submission_inputs: Iterable[SubmissionInput],
    truth: pd.DataFrame,
    weight_grid: Iterable[tuple[float, ...]],
    output_dir: Path,
    template: pd.DataFrame | None = None,
    class_policies: Iterable[str] = ("weighted-vote",),
    timestamp_tolerance_s: float = 1.0e-6,
):
    """Write a complete grid even when callers provide one-shot iterators."""

    return _ORIGINAL_WRITE(
        submission_inputs=tuple(submission_inputs),
        truth=truth,
        weight_grid=_materialize_weight_grid(weight_grid),
        output_dir=output_dir,
        template=template,
        class_policies=tuple(class_policies),
        timestamp_tolerance_s=timestamp_tolerance_s,
    )


_IMPL.evaluate_submission_ensemble_weight_grid = evaluate_submission_ensemble_weight_grid
_IMPL.write_submission_ensemble_weight_grid_outputs = (
    write_submission_ensemble_weight_grid_outputs
)

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
globals()["write_submission_ensemble_weight_grid_outputs"] = (
    write_submission_ensemble_weight_grid_outputs
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
