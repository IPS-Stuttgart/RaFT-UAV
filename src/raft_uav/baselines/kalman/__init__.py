"""Compatibility validation for Kalman tracking measurements.

The maintained implementation lives in the sibling ``kalman.py`` module. This
package preserves the public import path while rejecting asymmetric or indefinite
measurement covariances before they reach Kalman updates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from pyrecest.numerics import is_positive_semidefinite, is_symmetric

from raft_uav.numeric import optional_float

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
_ORIGINAL_WHITE_ACCELERATION_PROCESS_NOISE = _IMPL.white_acceleration_process_noise


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


def _nonnegative_finite_real(value: object, *, name: str) -> float:
    """Return a finite non-negative real scalar without Boolean coercion."""

    number = optional_float(value)
    if number is None or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return number


def white_acceleration_process_noise(
    dt_s: object,
    acceleration_std: object,
) -> np.ndarray:
    """Return valid white-acceleration covariance for non-negative controls."""

    dt = _nonnegative_finite_real(dt_s, name="dt_s")
    std = _nonnegative_finite_real(
        acceleration_std,
        name="acceleration_std",
    )
    return _ORIGINAL_WHITE_ACCELERATION_PROCESS_NOISE(dt, std)


_IMPL.TrackingMeasurement.__post_init__ = (
    _validated_tracking_measurement_post_init
)
_IMPL.white_acceleration_process_noise = white_acceleration_process_noise

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
globals()["_ORIGINAL_WHITE_ACCELERATION_PROCESS_NOISE"] = (
    _ORIGINAL_WHITE_ACCELERATION_PROCESS_NOISE
)
globals()["_validated_tracking_measurement_post_init"] = (
    _validated_tracking_measurement_post_init
)
globals()["_nonnegative_finite_real"] = _nonnegative_finite_real
globals()["white_acceleration_process_noise"] = white_acceleration_process_noise

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
