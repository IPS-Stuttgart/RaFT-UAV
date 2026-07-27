"""Measurement converters that consume learned heteroscedastic covariance columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.numeric import optional_float
from raft_uav.uncertainty import covariance_from_row


def rf_measurements_to_enu_with_uncertainty(
    rf: pd.DataFrame,
    *,
    default_std_m: float = 75.0,
) -> list[TrackingMeasurement]:
    """Convert normalized RF rows to measurements using row-wise covariance when present.

    The function is intended for frames that have already been normalized by
    :func:`raft_uav.io.aerpaw.normalize_rf` and optionally augmented with
    ``HeteroscedasticUncertaintyModel.apply_rf``.  It prefers learned
    ``cov_*`` columns, falls back to association covariance columns when those
    are present, and otherwise uses the historical CEP/default-std covariance.
    """

    default_std = _require_positive_float(default_std_m, name="default_std_m")
    measurements: list[TrackingMeasurement] = []
    for position, (_, row) in enumerate(rf.iterrows()):
        std_value = rf["std_m"].iloc[position] if "std_m" in rf.columns else None
        std_m = _positive_float(std_value) or default_std
        fallback = np.diag([std_m**2, std_m**2])
        measurements.append(
            TrackingMeasurement(
                time_s=_finite_real_scalar(
                    rf["time_s"].iloc[position],
                    name="time_s",
                ),
                vector=np.array(
                    [
                        _finite_real_scalar(
                            rf["east_m"].iloc[position],
                            name="east_m",
                        ),
                        _finite_real_scalar(
                            rf["north_m"].iloc[position],
                            name="north_m",
                        ),
                    ]
                ),
                covariance=covariance_from_row(
                    row,
                    2,
                    fallback,
                    prefixes=("cov", "association_cov"),
                ),
                source="rf",
            )
        )
    return measurements


def radar_measurements_to_enu_with_uncertainty(
    radar: pd.DataFrame,
    *,
    default_xy_std_m: float = 25.0,
    default_z_std_m: float = 35.0,
    default_velocity_std_mps: float = 12.0,
) -> list[TrackingMeasurement]:
    """Convert normalized radar rows to measurements using row-wise covariance.

    If Fortem velocity components are available, the returned measurement is
    six-dimensional.  The learned covariance is applied to the position block
    and the historical fixed velocity covariance is retained for the velocity
    block.
    """

    default_xy_std = _require_positive_float(default_xy_std_m, name="default_xy_std_m")
    default_z_std = _require_positive_float(default_z_std_m, name="default_z_std_m")
    default_velocity_std = _require_positive_float(
        default_velocity_std_mps,
        name="default_velocity_std_mps",
    )
    position_fallback = np.diag([default_xy_std**2, default_xy_std**2, default_z_std**2])
    measurements: list[TrackingMeasurement] = []
    for position_index, (_, row) in enumerate(radar.iterrows()):
        position = np.array(
            [
                _finite_real_scalar(
                    radar["east_m"].iloc[position_index],
                    name="east_m",
                ),
                _finite_real_scalar(
                    radar["north_m"].iloc[position_index],
                    name="north_m",
                ),
                _finite_real_scalar(
                    radar["up_m"].iloc[position_index],
                    name="up_m",
                ),
            ]
        )
        position_covariance = covariance_from_row(
            row,
            3,
            position_fallback,
            prefixes=("cov", "association_cov"),
        )
        velocity = _radar_velocity_vector_enu(row)
        if velocity is None:
            vector = position
            covariance = position_covariance
        else:
            vector = np.concatenate([position, velocity])
            covariance = np.zeros((6, 6), dtype=float)
            covariance[:3, :3] = position_covariance
            covariance[3:, 3:] = np.diag([default_velocity_std**2] * 3)
        measurements.append(
            TrackingMeasurement(
                time_s=_finite_real_scalar(
                    radar["time_s"].iloc[position_index],
                    name="time_s",
                ),
                vector=vector,
                covariance=covariance,
                source="radar",
            )
        )
    return measurements


def _radar_velocity_vector_enu(row: pd.Series) -> np.ndarray | None:
    required = ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps")
    if not all(column in row.index for column in required):
        return None
    try:
        velocity = np.array(
            [
                _finite_real_scalar(
                    row["velocity_east_mps"],
                    name="velocity_east_mps",
                ),
                _finite_real_scalar(
                    row["velocity_north_mps"],
                    name="velocity_north_mps",
                ),
                -_finite_real_scalar(
                    row["velocity_down_mps"],
                    name="velocity_down_mps",
                ),
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    return velocity if np.isfinite(velocity).all() else None


def _finite_real_scalar(value: object, *, name: str) -> float:
    """Return one finite real scalar without lossy NumPy coercion."""

    number = optional_float(value)
    if number is None:
        raise ValueError(f"{name} must be a finite real scalar")
    return number


def _require_positive_float(value: object, *, name: str) -> float:
    number = _finite_real_scalar(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


def _positive_float(value: object) -> float | None:
    try:
        number = _finite_real_scalar(value, name="value")
    except ValueError:
        return None
    return number if number > 0.0 else None
