"""Compatibility wrapper for Track 5 trajectory regularization.

The maintained implementation lives in the sibling
``track5_trajectory_regularizer.py`` module. This package preserves the public
import path while ensuring that reported robust weights correspond to the final
smoothed trajectory, repeated physical timestamps do not create ill-conditioned
acceleration penalties, and malformed numerical controls fail closed before
output artifacts are created.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_trajectory_regularizer.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_trajectory_regularizer_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        f"cannot load Track 5 trajectory-regularizer implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_ROBUST_SMOOTH_SEQUENCE = _IMPL._robust_smooth_sequence
_ORIGINAL_SECOND_DERIVATIVE_MATRIX = _IMPL._second_derivative_matrix
_ORIGINAL_REGULARIZE_TRACK5_ESTIMATES = _IMPL.regularize_track5_estimates
_ORIGINAL_RUN_TRACK5_TRAJECTORY_REGULARIZER = (
    _IMPL.run_track5_trajectory_regularizer
)


def _nonnegative_real_scalar(value: Any, *, field: str) -> float:
    """Return a finite non-negative real scalar."""

    numeric = optional_float(value)
    if numeric is None or numeric < 0.0:
        raise ValueError(f"{field} must be a finite non-negative real scalar")
    return numeric


def _positive_real_scalar(value: Any, *, field: str) -> float:
    """Return a finite positive real scalar."""

    numeric = optional_float(value)
    if numeric is None or numeric <= 0.0:
        raise ValueError(f"{field} must be a finite positive real scalar")
    return numeric


def _positive_integer_scalar(value: Any, *, field: str) -> int:
    """Return a finite positive integer-equivalent scalar without truncation."""

    numeric = optional_int(value)
    if numeric is None or numeric < 1:
        raise ValueError(f"{field} must be a positive finite integer")
    return numeric


def _validated_controls(
    *,
    smoothness_weight: Any,
    huber_delta_m: Any,
    iterations: Any,
    observation_sigma_m: Any,
) -> tuple[float, float, int, float]:
    """Normalize all regularizer controls under their documented constraints."""

    return (
        _nonnegative_real_scalar(smoothness_weight, field="smoothness_weight"),
        _positive_real_scalar(huber_delta_m, field="huber_delta_m"),
        _positive_integer_scalar(iterations, field="iterations"),
        _positive_real_scalar(observation_sigma_m, field="observation_sigma_m"),
    )


def _robust_smooth_sequence(
    times: np.ndarray,
    observed: np.ndarray,
    *,
    finite: np.ndarray,
    smoothness_weight: float,
    huber_delta_m: float,
    iterations: int,
    observation_sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return final-state residuals and Huber weights from the same iterate."""

    state, residual, _ = _ORIGINAL_ROBUST_SMOOTH_SEQUENCE(
        times,
        observed,
        finite=finite,
        smoothness_weight=smoothness_weight,
        huber_delta_m=huber_delta_m,
        iterations=iterations,
        observation_sigma_m=observation_sigma_m,
    )
    final_weights = np.zeros(len(residual), dtype=float)
    finite_mask = np.asarray(finite, dtype=bool)
    final_residual = np.asarray(residual, dtype=float)[finite_mask]
    final_weights[finite_mask] = np.minimum(
        1.0,
        float(huber_delta_m) / np.maximum(final_residual, 1.0e-12),
    )
    return state, residual, final_weights


def _second_derivative_matrix(times: np.ndarray) -> np.ndarray:
    """Build acceleration penalties on unique physical timestamps."""

    time_values = np.asarray(times, dtype=float)
    unique_times, inverse = np.unique(time_values, return_inverse=True)
    if len(unique_times) == len(time_values):
        return _ORIGINAL_SECOND_DERIVATIVE_MATRIX(time_values)
    if len(unique_times) < 3:
        return np.zeros((0, len(time_values)), dtype=float)

    # Replacing a zero interval by 1e-6 s makes duplicate template rows behave
    # like distinct physical samples one microsecond apart. Smooth the average
    # state at each physical timestamp instead while retaining every output row.
    unique_operator = _ORIGINAL_SECOND_DERIVATIVE_MATRIX(unique_times)
    averaging = np.zeros((len(unique_times), len(time_values)), dtype=float)
    counts = np.bincount(inverse, minlength=len(unique_times)).astype(float)
    columns = np.arange(len(time_values), dtype=int)
    averaging[inverse, columns] = 1.0 / counts[inverse]
    return unique_operator @ averaging


def regularize_track5_estimates(
    estimates: Any,
    *,
    smoothness_weight: Any = 10.0,
    huber_delta_m: Any = 25.0,
    iterations: Any = 5,
    observation_sigma_m: Any = 10.0,
):
    """Regularize estimates after strict, lossless control validation."""

    smoothness, huber, iteration_count, sigma = _validated_controls(
        smoothness_weight=smoothness_weight,
        huber_delta_m=huber_delta_m,
        iterations=iterations,
        observation_sigma_m=observation_sigma_m,
    )
    return _ORIGINAL_REGULARIZE_TRACK5_ESTIMATES(
        estimates,
        smoothness_weight=smoothness,
        huber_delta_m=huber,
        iterations=iteration_count,
        observation_sigma_m=sigma,
    )


def run_track5_trajectory_regularizer(
    *,
    estimates: Any,
    template: Any,
    output_dir: Any,
    class_map: Any = None,
    default_classification: Any = 0,
    max_nearest_time_delta_s: Any = None,
    resample_method: Any = "linear",
    max_interpolation_gap_s: Any = None,
    smoothness_weight: Any = 10.0,
    huber_delta_m: Any = 25.0,
    iterations: Any = 5,
    observation_sigma_m: Any = 10.0,
):
    """Run the regularizer after validating controls before output mutation."""

    smoothness, huber, iteration_count, sigma = _validated_controls(
        smoothness_weight=smoothness_weight,
        huber_delta_m=huber_delta_m,
        iterations=iterations,
        observation_sigma_m=observation_sigma_m,
    )
    return _ORIGINAL_RUN_TRACK5_TRAJECTORY_REGULARIZER(
        estimates=estimates,
        template=template,
        output_dir=output_dir,
        class_map=class_map,
        default_classification=default_classification,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
        smoothness_weight=smoothness,
        huber_delta_m=huber,
        iterations=iteration_count,
        observation_sigma_m=sigma,
    )


_IMPL._robust_smooth_sequence = _robust_smooth_sequence
_IMPL._second_derivative_matrix = _second_derivative_matrix
_IMPL.regularize_track5_estimates = regularize_track5_estimates
_IMPL.run_track5_trajectory_regularizer = run_track5_trajectory_regularizer

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_ROBUST_SMOOTH_SEQUENCE"] = _ORIGINAL_ROBUST_SMOOTH_SEQUENCE
globals()["_ORIGINAL_SECOND_DERIVATIVE_MATRIX"] = _ORIGINAL_SECOND_DERIVATIVE_MATRIX
globals()["_ORIGINAL_REGULARIZE_TRACK5_ESTIMATES"] = (
    _ORIGINAL_REGULARIZE_TRACK5_ESTIMATES
)
globals()["_ORIGINAL_RUN_TRACK5_TRAJECTORY_REGULARIZER"] = (
    _ORIGINAL_RUN_TRACK5_TRAJECTORY_REGULARIZER
)
globals()["_nonnegative_real_scalar"] = _nonnegative_real_scalar
globals()["_positive_real_scalar"] = _positive_real_scalar
globals()["_positive_integer_scalar"] = _positive_integer_scalar
globals()["_validated_controls"] = _validated_controls
globals()["_robust_smooth_sequence"] = _robust_smooth_sequence
globals()["_second_derivative_matrix"] = _second_derivative_matrix
globals()["regularize_track5_estimates"] = regularize_track5_estimates
globals()["run_track5_trajectory_regularizer"] = run_track5_trajectory_regularizer

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
