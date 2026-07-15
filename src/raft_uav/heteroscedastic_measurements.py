"""Measurement converters that consume learned heteroscedastic covariance columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
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
    for _, row in rf.iterrows():
        std_m = _positive_float(row.get("std_m")) or default_std
        fallback = np.diag([std_m**2, std_m**2])
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=np.array([float(row["east_m"]), float(row["north_m"])]),
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
    for _, row in radar.iterrows():
        position = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
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
                time_s=float(row["time_s"]),
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
                float(row["velocity_east_mps"]),
                float(row["velocity_north_mps"]),
                -float(row["velocity_down_mps"]),
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    return velocity if np.isfinite(velocity).all() else None


def _require_positive_float(value: object, *, name: str) -> float:
    number = _positive_float(value)
    if number is None:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool | np.bool_):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0.0 else None
