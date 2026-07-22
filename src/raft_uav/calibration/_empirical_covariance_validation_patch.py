"""Strict scalar validation for empirical covariance controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from raft_uav.calibration import empirical_covariance as _IMPL

_ORIGINAL_ESTIMATE_EMPIRICAL_MEASUREMENT_COVARIANCES = (
    _IMPL.estimate_empirical_measurement_covariances
)
_ORIGINAL_ALIGNED_RESIDUALS = _IMPL.aligned_residuals
_ORIGINAL_EMPIRICAL_COVARIANCE_MATRIX = _IMPL.empirical_covariance_matrix


def _validated_real_scalar(
    value: object,
    *,
    field_name: str,
    requirement: str,
) -> float:
    error = f"{field_name} must be {requirement}"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(error)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(error)
    try:
        number = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number):
        raise ValueError(error)
    return number


def _nonnegative_time_gate(value: object) -> float:
    gate = _validated_real_scalar(
        value,
        field_name="max_time_delta_s",
        requirement="a finite non-negative real scalar",
    )
    if gate < 0.0:
        raise ValueError(
            "max_time_delta_s must be a finite non-negative real scalar"
        )
    return gate


def _positive_variance_floor(value: object) -> float:
    floor = _validated_real_scalar(
        value,
        field_name="min_variance_m2",
        requirement="a positive finite real scalar",
    )
    if floor <= 0.0:
        raise ValueError("min_variance_m2 must be a positive finite real scalar")
    return floor


def estimate_empirical_measurement_covariances(
    *,
    rf: pd.DataFrame | None,
    radar: pd.DataFrame | None,
    truth: pd.DataFrame,
    max_time_delta_s: float = 2.0,
    min_variance_m2: float = 1.0,
) -> dict[str, Any]:
    """Estimate empirical covariances after validating scalar controls."""

    return _ORIGINAL_ESTIMATE_EMPIRICAL_MEASUREMENT_COVARIANCES(
        rf=rf,
        radar=radar,
        truth=truth,
        max_time_delta_s=_nonnegative_time_gate(max_time_delta_s),
        min_variance_m2=_positive_variance_floor(min_variance_m2),
    )


def aligned_residuals(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    max_time_delta_s: float,
) -> np.ndarray:
    """Align residuals after validating the nearest-time gate."""

    return _ORIGINAL_ALIGNED_RESIDUALS(
        frame,
        truth,
        source=source,
        max_time_delta_s=_nonnegative_time_gate(max_time_delta_s),
    )


def empirical_covariance_matrix(
    residuals: np.ndarray,
    *,
    min_variance_m2: float = 1.0,
) -> np.ndarray:
    """Estimate a covariance after validating its diagonal variance floor."""

    return _ORIGINAL_EMPIRICAL_COVARIANCE_MATRIX(
        residuals,
        min_variance_m2=_positive_variance_floor(min_variance_m2),
    )


_IMPL.estimate_empirical_measurement_covariances = (
    estimate_empirical_measurement_covariances
)
_IMPL.aligned_residuals = aligned_residuals
_IMPL.empirical_covariance_matrix = empirical_covariance_matrix
