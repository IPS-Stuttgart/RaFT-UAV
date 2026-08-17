"""Validate covariance geometry at the public tracking-measurement boundary."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_raft_uav_tracking_measurement_covariance_validation_patch_applied"


def _validate_covariance_geometry(value: Any, *, label: str) -> None:
    """Reject finite square matrices that are not valid covariance matrices."""

    try:
        covariance = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        # The wrapped implementation owns type/conversion validation.
        return
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        # The wrapped implementation owns dimension validation.
        return
    if covariance.size == 0 or not np.isfinite(covariance).all():
        # The wrapped implementation owns empty/non-finite validation.
        return

    dimension = max(1, covariance.shape[0])
    scale = max(1.0, float(np.max(np.abs(covariance))))
    eps = np.finfo(float).eps
    symmetry_tolerance = 64.0 * eps * dimension * scale
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=0.0,
        atol=symmetry_tolerance,
    ):
        raise ValueError(f"{label} covariance must be symmetric")

    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    psd_tolerance = 256.0 * eps * dimension * scale
    if float(eigenvalues[0]) < -psd_tolerance:
        raise ValueError(f"{label} covariance must be positive semidefinite")


def apply_tracking_measurement_covariance_validation_patch(kalman_module: Any) -> None:
    """Guard raw and runtime-calibrated measurement covariance matrices."""

    if getattr(kalman_module, _PATCH_MARKER, False):
        return

    original_post_init = kalman_module.TrackingMeasurement.__post_init__

    @wraps(original_post_init)
    def tracking_measurement_post_init(
        self: Any,
        _apply_runtime_calibration: bool,
    ) -> None:
        _validate_covariance_geometry(self.covariance, label="measurement")
        original_post_init(self, _apply_runtime_calibration)
        _validate_covariance_geometry(
            self.covariance,
            label="calibrated measurement",
        )

    kalman_module.TrackingMeasurement.__post_init__ = tracking_measurement_post_init
    setattr(kalman_module, _PATCH_MARKER, True)
