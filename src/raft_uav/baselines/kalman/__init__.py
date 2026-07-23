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


_IMPL.TrackingMeasurement.__post_init__ = (
    _validated_tracking_measurement_post_init
)

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
globals()["_validated_tracking_measurement_post_init"] = (
    _validated_tracking_measurement_post_init
)

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
