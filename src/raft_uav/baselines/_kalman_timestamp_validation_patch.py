"""Runtime validation for asynchronous Kalman and IMM tracker scalar inputs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


_kalman = import_module("raft_uav.baselines.kalman")
_imm = import_module("raft_uav.baselines.imm")
_ORIGINAL_TRACKING_MEASUREMENT_POST_INIT = _kalman.TrackingMeasurement.__post_init__
_ORIGINAL_TRACKER_INIT = _kalman.AsyncConstantVelocityKalmanTracker.__init__
_ORIGINAL_IS_BOOTSTRAP_MEASUREMENT = (
    _kalman.AsyncConstantVelocityKalmanTracker._is_bootstrap_measurement
)
_ORIGINAL_PREDICT_TO = _kalman.AsyncConstantVelocityKalmanTracker.predict_to
_ORIGINAL_COAST_TO = _kalman.AsyncConstantVelocityKalmanTracker.coast_to
_ORIGINAL_IMM_TRACKER_INIT = _imm.AsyncInteractingMultipleModelTracker.__init__
_ORIGINAL_IMM_PREDICT_TO = _imm.AsyncInteractingMultipleModelTracker.predict_to


def _finite_timestamp_seconds(value: Any, *, field_name: str) -> float:
    """Return a finite scalar timestamp or raise a field-specific error."""

    error = f"{field_name} must be a finite numeric timestamp"
    if isinstance(value, bool | np.bool_) or np.ma.is_masked(value):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        timestamp_s = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(timestamp_s):
        raise ValueError(error)
    return timestamp_s


def _finite_nonnegative_scale(value: Any, *, field_name: str) -> float:
    """Return a finite nonnegative scalar uncertainty scale."""

    error = f"{field_name} must be a finite nonnegative scalar"
    if isinstance(value, bool | np.bool_) or np.ma.is_masked(value):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0:
        raise ValueError(error)
    try:
        scale = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(error)
    return scale


def _boolean_scalar_hidden_in_arrays(value: Any) -> bool:
    """Return whether zero-dimensional array wrappers contain a Boolean scalar."""

    seen_array_ids: set[int] = set()
    while isinstance(value, np.ndarray) and value.ndim == 0:
        array_id = id(value)
        if array_id in seen_array_ids:
            return False
        seen_array_ids.add(array_id)
        value = value.item()
    return isinstance(value, bool | np.bool_)


def _reject_boolean_values(value: Any, *, field_name: str) -> None:
    """Reject Boolean pseudo-numbers before float coercion rewrites them as 0/1."""

    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return
    if any(_boolean_scalar_hidden_in_arrays(item) for item in array.flat):
        raise ValueError(f"{field_name} must contain only real non-Boolean values")


def _finite_initial_position(value: Any) -> np.ndarray:
    """Return a finite real 2D, 3D, or 6D initial position/state vector."""

    error = "initial_position must contain 2, 3, or 6 finite real values"
    _reject_boolean_values(value, field_name="initial_position")
    try:
        masked = np.ma.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if bool(np.any(np.ma.getmaskarray(masked))) or np.iscomplexobj(masked.data):
        raise ValueError(error)
    try:
        position = np.asarray(masked.data, dtype=float).reshape(-1)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if position.size not in (2, 3, 6) or not np.isfinite(position).all():
        raise ValueError(error)
    return position


def _reject_masked_values(value: Any, *, field_name: str) -> None:
    """Reject arrays whose mask would be discarded by ``np.asarray``."""

    if np.ma.isMaskedArray(value) and bool(np.any(np.ma.getmaskarray(value))):
        raise ValueError(f"{field_name} must not contain masked values")


def _reject_complex_values(value: Any, *, field_name: str) -> None:
    """Reject arrays whose imaginary components would be discarded on coercion."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return
    if np.iscomplexobj(array) or (
        array.dtype == object and any(np.iscomplexobj(item) for item in array.flat)
    ):
        raise ValueError(f"{field_name} must contain only real values")


def _tracking_measurement_post_init(
    self: Any,
    _apply_runtime_calibration: bool,
) -> None:
    _reject_masked_values(self.vector, field_name="measurement vector")
    _reject_masked_values(self.covariance, field_name="measurement covariance")
    _reject_boolean_values(self.vector, field_name="measurement vector")
    _reject_boolean_values(self.covariance, field_name="measurement covariance")
    _reject_complex_values(self.vector, field_name="measurement vector")
    _reject_complex_values(self.covariance, field_name="measurement covariance")
    time_s = _finite_timestamp_seconds(self.time_s, field_name="measurement time_s")
    _ORIGINAL_TRACKING_MEASUREMENT_POST_INIT(self, _apply_runtime_calibration)
    object.__setattr__(self, "time_s", time_s)


def _tracker_init(
    self: Any,
    initial_position: np.ndarray,
    initial_time_s: float,
    initial_position_std_m: float = 50.0,
    initial_velocity_std_mps: float = 15.0,
    acceleration_std_mps2: float = 4.0,
) -> None:
    validated_initial_position = _finite_initial_position(initial_position)
    validated_time_s = _finite_timestamp_seconds(
        initial_time_s,
        field_name="initial_time_s",
    )
    validated_position_std_m = _finite_nonnegative_scale(
        initial_position_std_m,
        field_name="initial_position_std_m",
    )
    validated_velocity_std_mps = _finite_nonnegative_scale(
        initial_velocity_std_mps,
        field_name="initial_velocity_std_mps",
    )
    validated_acceleration_std_mps2 = _finite_nonnegative_scale(
        acceleration_std_mps2,
        field_name="acceleration_std_mps2",
    )
    _ORIGINAL_TRACKER_INIT(
        self,
        validated_initial_position,
        validated_time_s,
        initial_position_std_m=validated_position_std_m,
        initial_velocity_std_mps=validated_velocity_std_mps,
        acceleration_std_mps2=validated_acceleration_std_mps2,
    )


def _is_bootstrap_measurement(self: Any, measurement: Any) -> bool:
    """Require an absolute timestamp match for bootstrap suppression."""

    if not self._initial_update_pending:
        return False
    if not np.isclose(
        float(measurement.time_s),
        float(self.current_time_s),
        rtol=0.0,
        atol=1.0e-9,
    ):
        return False
    return bool(_ORIGINAL_IS_BOOTSTRAP_MEASUREMENT(self, measurement))


def _predict_to(self: Any, time_s: float) -> None:
    validated_time_s = _finite_timestamp_seconds(time_s, field_name="time_s")
    _ORIGINAL_PREDICT_TO(self, validated_time_s)


def _coast_to(self: Any, time_s: float) -> None:
    """Validate before coast bookkeeping consumes bootstrap state."""

    validated_time_s = _finite_timestamp_seconds(time_s, field_name="time_s")
    if validated_time_s < self.current_time_s - 1.0e-9:
        raise ValueError("measurements must be processed in chronological order")
    _ORIGINAL_COAST_TO(self, validated_time_s)


def _imm_tracker_init(
    self: Any,
    initial_position: np.ndarray,
    initial_time_s: float,
    initial_position_std_m: float = 50.0,
    initial_velocity_std_mps: float = 15.0,
    acceleration_std_mps2: float = 4.0,
    modes: Any = None,
    initial_mode_probabilities: Any = None,
    mode_switch_time_constant_s: float = 20.0,
) -> None:
    validated_initial_position = _finite_initial_position(initial_position)
    validated_time_s = _finite_timestamp_seconds(
        initial_time_s,
        field_name="initial_time_s",
    )
    validated_position_std_m = _finite_nonnegative_scale(
        initial_position_std_m,
        field_name="initial_position_std_m",
    )
    validated_velocity_std_mps = _finite_nonnegative_scale(
        initial_velocity_std_mps,
        field_name="initial_velocity_std_mps",
    )
    validated_acceleration_std_mps2 = _finite_nonnegative_scale(
        acceleration_std_mps2,
        field_name="acceleration_std_mps2",
    )
    _ORIGINAL_IMM_TRACKER_INIT(
        self,
        validated_initial_position,
        validated_time_s,
        initial_position_std_m=validated_position_std_m,
        initial_velocity_std_mps=validated_velocity_std_mps,
        acceleration_std_mps2=validated_acceleration_std_mps2,
        modes=modes,
        initial_mode_probabilities=initial_mode_probabilities,
        mode_switch_time_constant_s=mode_switch_time_constant_s,
    )


def _imm_predict_to(self: Any, time_s: float) -> None:
    validated_time_s = _finite_timestamp_seconds(time_s, field_name="time_s")
    _ORIGINAL_IMM_PREDICT_TO(self, validated_time_s)


def apply_kalman_timestamp_validation_patch() -> None:
    """Install scalar validation at public asynchronous tracker boundaries."""

    if not getattr(_kalman, "_timestamp_validation_patch_applied", False):
        _kalman.TrackingMeasurement.__post_init__ = _tracking_measurement_post_init
        _kalman.AsyncConstantVelocityKalmanTracker.__init__ = _tracker_init
        _kalman.AsyncConstantVelocityKalmanTracker._is_bootstrap_measurement = (
            _is_bootstrap_measurement
        )
        _kalman.AsyncConstantVelocityKalmanTracker.predict_to = _predict_to
        _kalman.AsyncConstantVelocityKalmanTracker.coast_to = _coast_to
        _kalman._timestamp_validation_patch_applied = True

    if not getattr(_imm, "_timestamp_validation_patch_applied", False):
        _imm.AsyncInteractingMultipleModelTracker.__init__ = _imm_tracker_init
        _imm.AsyncInteractingMultipleModelTracker.predict_to = _imm_predict_to
        _imm._timestamp_validation_patch_applied = True
