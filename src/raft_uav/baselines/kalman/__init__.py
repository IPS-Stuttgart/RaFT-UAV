"""Compatibility validation for Kalman tracking measurements.

The maintained implementation lives in the sibling ``kalman.py`` module. This
package preserves the public import path while rejecting asymmetric or indefinite
measurement covariances and invalid tracker initialization values before they
reach Kalman updates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import importlib.util
from pathlib import Path
import sys

import numpy as np
from pyrecest.numerics import is_positive_semidefinite, is_symmetric

_IMPL_PATH = Path(__file__).resolve().parent.parent / "kalman.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._kalman_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load Kalman implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRACKING_MEASUREMENT_POST_INIT = _IMPL.TrackingMeasurement.__post_init__
_ORIGINAL_TRACKER_INIT = _IMPL.AsyncConstantVelocityKalmanTracker.__init__
_ORIGINAL_RUN_ASYNC_CV_BASELINE = _IMPL.run_async_cv_baseline
_ORIGINAL_CONSTANT_VELOCITY_MATRIX = _IMPL.constant_velocity_matrix
_ORIGINAL_WHITE_ACCELERATION_PROCESS_NOISE = _IMPL.white_acceleration_process_noise
_SOURCE_PRIORITY = {"rf": 0, "radar": 1}


def _validated_tracking_measurement_post_init(
    self: object,
    _apply_runtime_calibration: bool,
) -> None:
    """Validate the effective covariance after optional runtime calibration."""

    _ORIGINAL_TRACKING_MEASUREMENT_POST_INIT(
        self,
        _apply_runtime_calibration,
    )
    covariance = np.asarray(self.covariance, dtype=float)
    if not is_symmetric(covariance):
        raise ValueError("measurement covariance must be symmetric")
    if not is_positive_semidefinite(covariance):
        raise ValueError("measurement covariance must be positive semidefinite")
    object.__setattr__(
        self,
        "covariance",
        0.5 * (covariance + covariance.T),
    )


def _finite_real_scalar(value: object, *, name: str, nonnegative: bool = False) -> float:
    """Return a finite scalar, optionally requiring a non-negative value."""

    parsed = _IMPL.optional_float(value)
    if parsed is None or (nonnegative and parsed < 0.0):
        qualifier = "finite, non-negative real scalar" if nonnegative else "finite real scalar"
        raise ValueError(f"{name} must be a {qualifier}")
    return parsed


def constant_velocity_matrix(dt_s: float) -> np.ndarray:
    """Return a CV transition matrix for a finite non-negative time step."""

    dt = _finite_real_scalar(dt_s, name="dt_s", nonnegative=True)
    return _ORIGINAL_CONSTANT_VELOCITY_MATRIX(dt)


def white_acceleration_process_noise(
    dt_s: float,
    acceleration_std: float,
) -> np.ndarray:
    """Return finite positive-semidefinite CV process noise."""

    dt = _finite_real_scalar(dt_s, name="dt_s", nonnegative=True)
    std = _finite_real_scalar(
        acceleration_std,
        name="acceleration_std",
        nonnegative=True,
    )
    return _ORIGINAL_WHITE_ACCELERATION_PROCESS_NOISE(dt, std)


def _finite_initial_position(value: object) -> np.ndarray:
    """Return a finite real 2D, 3D, or position-plus-velocity state vector."""

    message = "initial_position must contain 2, 3, or 6 finite real values"
    try:
        masked = np.ma.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if bool(np.ma.getmaskarray(masked).any()) or np.iscomplexobj(masked.data):
        raise ValueError(message)
    try:
        position = np.asarray(masked.data, dtype=float).reshape(-1)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if position.size not in (2, 3, 6) or not np.isfinite(position).all():
        raise ValueError(message)
    return position


def _validated_tracker_init(
    self: object,
    initial_position: np.ndarray,
    initial_time_s: float,
    initial_position_std_m: float = 50.0,
    initial_velocity_std_mps: float = 15.0,
    acceleration_std_mps2: float = 4.0,
) -> None:
    """Reject malformed initialization values before constructing filter state."""

    position = _finite_initial_position(initial_position)
    time_s = _finite_real_scalar(initial_time_s, name="initial_time_s")
    position_std = _finite_real_scalar(
        initial_position_std_m,
        name="initial_position_std_m",
        nonnegative=True,
    )
    velocity_std = _finite_real_scalar(
        initial_velocity_std_mps,
        name="initial_velocity_std_mps",
        nonnegative=True,
    )
    acceleration_std = _finite_real_scalar(
        acceleration_std_mps2,
        name="acceleration_std_mps2",
        nonnegative=True,
    )
    _ORIGINAL_TRACKER_INIT(
        self,
        position,
        time_s,
        position_std,
        velocity_std,
        acceleration_std,
    )


def _tracking_measurement_order_key(
    measurement: _IMPL.TrackingMeasurement,
) -> tuple[object, ...]:
    """Return a deterministic order for measurements sharing a timestamp."""

    source = str(measurement.source)
    normalized_source = source.casefold()
    vector = tuple(
        np.asarray(measurement.vector, dtype=float).reshape(-1).tolist()
    )
    covariance = tuple(
        np.asarray(measurement.covariance, dtype=float).reshape(-1).tolist()
    )
    return (
        float(measurement.time_s),
        _SOURCE_PRIORITY.get(normalized_source, len(_SOURCE_PRIORITY)),
        normalized_source,
        source,
        len(vector),
        vector,
        covariance,
    )


def run_async_cv_baseline(
    measurements: Iterable[_IMPL.TrackingMeasurement],
    acceleration_std_mps2: float = 4.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    student_t_dof_by_source: Mapping[str, float] | None = None,
    huber_threshold_by_source: Mapping[str, float] | None = None,
) -> list[dict[str, object]]:
    """Run the CV baseline with deterministic same-timestamp measurement order."""

    ordered = sorted(measurements, key=_tracking_measurement_order_key)
    return _ORIGINAL_RUN_ASYNC_CV_BASELINE(
        ordered,
        acceleration_std_mps2=acceleration_std_mps2,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
        student_t_dof_by_source=student_t_dof_by_source,
        huber_threshold_by_source=huber_threshold_by_source,
    )


_IMPL.TrackingMeasurement.__post_init__ = (
    _validated_tracking_measurement_post_init
)
_IMPL.AsyncConstantVelocityKalmanTracker.__init__ = _validated_tracker_init
_IMPL.constant_velocity_matrix = constant_velocity_matrix
_IMPL.white_acceleration_process_noise = white_acceleration_process_noise
_IMPL.run_async_cv_baseline = run_async_cv_baseline

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_TRACKING_MEASUREMENT_POST_INIT"] = (
    _ORIGINAL_TRACKING_MEASUREMENT_POST_INIT
)
globals()["_ORIGINAL_TRACKER_INIT"] = _ORIGINAL_TRACKER_INIT
globals()["_ORIGINAL_RUN_ASYNC_CV_BASELINE"] = _ORIGINAL_RUN_ASYNC_CV_BASELINE
globals()["_ORIGINAL_CONSTANT_VELOCITY_MATRIX"] = _ORIGINAL_CONSTANT_VELOCITY_MATRIX
globals()["_ORIGINAL_WHITE_ACCELERATION_PROCESS_NOISE"] = (
    _ORIGINAL_WHITE_ACCELERATION_PROCESS_NOISE
)
globals()["_SOURCE_PRIORITY"] = _SOURCE_PRIORITY
globals()["_validated_tracking_measurement_post_init"] = (
    _validated_tracking_measurement_post_init
)
globals()["_finite_real_scalar"] = _finite_real_scalar
globals()["constant_velocity_matrix"] = constant_velocity_matrix
globals()["white_acceleration_process_noise"] = white_acceleration_process_noise
globals()["_finite_initial_position"] = _finite_initial_position
globals()["_validated_tracker_init"] = _validated_tracker_init
globals()["_tracking_measurement_order_key"] = _tracking_measurement_order_key
globals()["run_async_cv_baseline"] = run_async_cv_baseline

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
