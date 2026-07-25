"""Compatibility validation for class-probability calibration metrics.

The maintained implementation lives in the sibling
``class_probability_calibration.py`` module. This package preserves the public
import path while rejecting malformed truth-class indices before metric
calculation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from raft_uav.numeric import optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "class_probability_calibration.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._class_probability_calibration_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        "cannot load class-probability calibration implementation from "
        f"{_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_CLASSIFICATION_CALIBRATION_METRICS = (
    _IMPL.classification_calibration_metrics
)
_ORIGINAL_MULTICLASS_NLL = _IMPL.multiclass_nll
_ORIGINAL_EXPECTED_CALIBRATION_ERROR = _IMPL.expected_calibration_error


def _validated_metric_inputs(
    probabilities: Any,
    truth_indices: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return metric arrays with exact in-range class indices."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional multi-class matrix")

    truth_values = np.ma.asarray(truth_indices, dtype=object)
    if truth_values.ndim != 1:
        raise ValueError("truth_indices must be one-dimensional")
    if len(truth_values) != len(values):
        raise ValueError(
            "truth_indices must contain one class index per probability row"
        )

    truth = np.empty(len(truth_values), dtype=int)
    invalid_positions: list[int] = []
    for position, value in enumerate(truth_values.tolist()):
        class_index = optional_int(value)
        if class_index is None or class_index < 0 or class_index >= values.shape[1]:
            invalid_positions.append(position)
        else:
            truth[position] = class_index

    if invalid_positions:
        raise ValueError(
            "truth_indices must contain exact class indices in "
            f"[0, {values.shape[1] - 1}]; invalid rows: {invalid_positions}"
        )
    return values, truth


def classification_calibration_metrics(
    probabilities: np.ndarray,
    truth_indices: np.ndarray,
    *,
    ece_bins: int = 10,
    epsilon: float = 1.0e-9,
) -> dict[str, float]:
    """Return calibration metrics after validating truth-class indices."""

    values, truth = _validated_metric_inputs(probabilities, truth_indices)
    return _ORIGINAL_CLASSIFICATION_CALIBRATION_METRICS(
        values,
        truth,
        ece_bins=ece_bins,
        epsilon=epsilon,
    )


def multiclass_nll(
    probabilities: np.ndarray,
    truth_indices: np.ndarray,
    *,
    epsilon: float = 1.0e-9,
) -> float:
    """Return multiclass NLL after validating truth-class indices."""

    values, truth = _validated_metric_inputs(probabilities, truth_indices)
    return _ORIGINAL_MULTICLASS_NLL(values, truth, epsilon=epsilon)


def expected_calibration_error(
    probabilities: np.ndarray,
    truth_indices: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Return expected calibration error for valid truth-class indices."""

    values, truth = _validated_metric_inputs(probabilities, truth_indices)
    return _ORIGINAL_EXPECTED_CALIBRATION_ERROR(values, truth, bins=bins)


_IMPL.classification_calibration_metrics = classification_calibration_metrics
_IMPL.multiclass_nll = multiclass_nll
_IMPL.expected_calibration_error = expected_calibration_error

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_metric_inputs"] = _validated_metric_inputs
globals()["classification_calibration_metrics"] = classification_calibration_metrics
globals()["multiclass_nll"] = multiclass_nll
globals()["expected_calibration_error"] = expected_calibration_error

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_IMPL.main())
