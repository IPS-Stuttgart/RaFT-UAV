"""Validate effective covariance matrices at the tracking-measurement boundary."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


_kalman = import_module("raft_uav.baselines.kalman")
_ORIGINAL_TRACKING_MEASUREMENT_POST_INIT = _kalman.TrackingMeasurement.__post_init__


def _validated_covariance(value: Any) -> np.ndarray:
    """Return an exactly symmetric PSD covariance or raise a clear error."""

    covariance = np.asarray(value, dtype=float)
    symmetric = 0.5 * (covariance + covariance.T)
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=1.0e-10,
        atol=1.0e-12 * scale,
    ):
        raise ValueError("measurement covariance must be symmetric")

    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "measurement covariance must be positive semidefinite"
        ) from exc
    eigenvalue_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -1.0e-12 * eigenvalue_scale:
        raise ValueError("measurement covariance must be positive semidefinite")
    return symmetric


def _tracking_measurement_post_init(
    self: Any,
    _apply_runtime_calibration: bool,
) -> None:
    """Validate the covariance produced by the complete calibration boundary."""

    _ORIGINAL_TRACKING_MEASUREMENT_POST_INIT(self, _apply_runtime_calibration)
    object.__setattr__(
        self,
        "covariance",
        _validated_covariance(self.covariance),
    )


def install() -> None:
    """Install effective-covariance validation once."""

    if getattr(_kalman, "_measurement_covariance_validation_patch_applied", False):
        return
    _kalman.TrackingMeasurement.__post_init__ = _tracking_measurement_post_init
    _kalman._measurement_covariance_validation_patch_applied = True
