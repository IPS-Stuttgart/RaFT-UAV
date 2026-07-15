"""Runtime validation for asynchronous Kalman tracker scalar inputs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


_kalman = import_module("raft_uav.baselines.kalman")
_ORIGINAL_TRACKING_MEASUREMENT_POST_INIT = _kalman.TrackingMeasurement.__post_init__
_ORIGINAL_TRACKER_INIT = _kalman.AsyncConstantVelocityKalmanTracker.__init__
_ORIGINAL_PREDICT_TO = _kalman.AsyncConstantVelocityKalmanTracker.predict_to


def _finite_timestamp_seconds(value: Any, *, field_name: str) -> float:
    """Return a finite scalar timestamp or raise a field-specific error."""

    error = f"{field_name} must be a finite numeric timestamp"
    if isinstance(value, bool | np.bool_):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0:
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
    if isinstance(value, bool | np.bool_):
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


def _tracking_measurement_post_init(
    self: Any,
    _apply_runtime_calibration: bool,
) -> None:
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
    validated_time_s = _finite_timestamp_seconds(initial_time_s, field_name="initial_time_s")
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
        initial_position,
        validated_time_s,
        initial_position_std_m=validated_position_std_m,
        initial_velocity_std_mps=validated_velocity_std_mps,
        acceleration_std_mps2=validated_acceleration_std_mps2,
    )


def _predict_to(self: Any, time_s: float) -> None:
    validated_time_s = _finite_timestamp_seconds(time_s, field_name="time_s")
    _ORIGINAL_PREDICT_TO(self, validated_time_s)


def apply_kalman_timestamp_validation_patch() -> None:
    """Install scalar validation at public asynchronous Kalman boundaries."""

    if getattr(_kalman, "_timestamp_validation_patch_applied", False):
        return
    _kalman.TrackingMeasurement.__post_init__ = _tracking_measurement_post_init
    _kalman.AsyncConstantVelocityKalmanTracker.__init__ = _tracker_init
    _kalman.AsyncConstantVelocityKalmanTracker.predict_to = _predict_to
    _kalman._timestamp_validation_patch_applied = True
