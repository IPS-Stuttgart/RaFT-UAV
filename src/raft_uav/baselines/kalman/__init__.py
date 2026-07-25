"""Compatibility validation for Kalman tracking measurements.

The maintained implementation lives in the sibling ``kalman.py`` module. This
package preserves the public import path while rejecting asymmetric or indefinite
measurement covariances, invalid tracker initialization values, and malformed
prediction timestamps before they reach Kalman updates.
"""

from __future__ import annotations

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
_ORIGINAL_TRACKER_PREDICT_TO = _IMPL.AsyncConstantVelocityKalmanTracker.predict_to


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
        qualifier = (
            "finite, non-negative real scalar"
            if nonnegative
            else "finite real scalar"
        )
        raise ValueError(f"{name} must be a {qualifier}")
    return parsed


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


def _validated_predict_to(self: object, time_s: float) -> None:
    """Reject malformed prediction timestamps before constructing dynamics."""

    target_time_s = _finite_real_scalar(time_s, name="time_s")
    _ORIGINAL_TRACKER_PREDICT_TO(self, target_time_s)


def _validated_coast_to(self: object, time_s: float) -> None:
    """Coast atomically so rejected timestamps do not consume bootstrap state."""

    target_time_s = _finite_real_scalar(time_s, name="time_s")
    _ORIGINAL_TRACKER_PREDICT_TO(self, target_time_s)
    self._initial_update_pending = False
    self._last_prior_mean = self.mean.copy()
    self._last_prior_covariance = self.covariance.copy()


_IMPL.TrackingMeasurement.__post_init__ = (
    _validated_tracking_measurement_post_init
)
_IMPL.AsyncConstantVelocityKalmanTracker.__init__ = _validated_tracker_init
_IMPL.AsyncConstantVelocityKalmanTracker.predict_to = _validated_predict_to
_IMPL.AsyncConstantVelocityKalmanTracker.coast_to = _validated_coast_to

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
globals()["_ORIGINAL_TRACKER_PREDICT_TO"] = _ORIGINAL_TRACKER_PREDICT_TO
globals()["_validated_tracking_measurement_post_init"] = (
    _validated_tracking_measurement_post_init
)
globals()["_finite_real_scalar"] = _finite_real_scalar
globals()["_finite_initial_position"] = _finite_initial_position
globals()["_validated_tracker_init"] = _validated_tracker_init
globals()["_validated_predict_to"] = _validated_predict_to
globals()["_validated_coast_to"] = _validated_coast_to

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
