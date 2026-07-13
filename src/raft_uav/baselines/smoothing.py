"""RTS, fixed-lag, and robust MAP smoothing for CV tracking records."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from pyrecest.smoothers import smooth_records as smooth_pyrecest_records

from raft_uav.baselines.kalman import (
    TrackingMeasurement,
    constant_velocity_matrix,
    white_acceleration_process_noise,
)
from raft_uav.baselines.record_helpers import copy_record
from raft_uav.baselines.robust_map import RobustMapSmootherConfig, robust_map_smooth_records

SMOOTHER_MODES = ("none", "rts", "fixed-lag", "robust-map", "fixed-lag-map")


def smooth_tracking_records(
    records: list[dict[str, object]],
    *,
    method: str,
    acceleration_std_mps2: float,
    lag_s: float | None = None,
    measurements: Iterable[TrackingMeasurement] | None = None,
    robust_map_config: RobustMapSmootherConfig | None = None,
) -> list[dict[str, object]]:
    """Return tracking records with smoothed state/covariance estimates.

    ``rts`` and ``fixed-lag`` delegate to PyRecEst's generic asynchronous record
    smoother. ``robust-map`` and ``fixed-lag-map`` remain RaFT-UAV-specific
    because they build an application-specific RF/radar measurement factor graph.
    """

    if method not in SMOOTHER_MODES:
        raise ValueError(f"unknown smoother method {method!r}")
    if method == "none" or not records:
        return [copy_record(record) for record in records]

    acceleration_std_mps2 = _finite_nonnegative(
        acceleration_std_mps2,
        name="acceleration_std_mps2",
    )
    validated_lag_s = None
    if method in ("fixed-lag", "fixed-lag-map"):
        if lag_s is None:
            raise ValueError(f"{method} smoothing requires a nonnegative lag_s")
        validated_lag_s = _finite_nonnegative(lag_s, name="lag_s")
    if method in ("robust-map", "fixed-lag-map"):
        return robust_map_smooth_records(
            records,
            measurements=measurements,
            acceleration_std_mps2=acceleration_std_mps2,
            config=robust_map_config,
            lag_s=None if method == "robust-map" else validated_lag_s,
        )

    return smooth_pyrecest_records(
        records,
        method="rts" if method == "rts" else "fixed-lag",
        lag=None if method == "rts" else validated_lag_s,
        transition_model=_constant_velocity_transition_for_state_dim,
        process_noise_model=lambda dt_s, state_dim: _constant_velocity_process_noise_for_state_dim(
            dt_s,
            state_dim,
            acceleration_std_mps2=acceleration_std_mps2,
        ),
        time_key="time_s",
        state_key="state",
        covariance_key="covariance",
        output_state_key="state",
        output_covariance_key="covariance",
        filtered_state_key="filtered_state",
        filtered_covariance_key="filtered_covariance",
        metadata={
            "smoother_method": method,
            "smoother_lag_s": None if method == "rts" else validated_lag_s,
        },
    )


def _finite_nonnegative(value: float, *, name: str) -> float:
    """Return a finite nonnegative smoothing control."""

    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def _constant_velocity_transition_for_state_dim(dt_s: float, state_dim: int) -> np.ndarray:
    """Return a CV transition embedded in a record's state dimension.

    The standard RaFT-UAV state is 6D ``[e,n,u,ve,vn,vu]``.  Experimental
    variants may append bias states; those extra dimensions are modeled as
    identity dynamics by the generic record smoother unless a specialized
    smoother is used.
    """

    base = constant_velocity_matrix(dt_s)
    return _embed_matrix(base, state_dim=state_dim, diagonal_for_extra=1.0)


def _constant_velocity_process_noise_for_state_dim(
    dt_s: float,
    state_dim: int,
    *,
    acceleration_std_mps2: float,
) -> np.ndarray:
    """Return CV process noise embedded in a record's state dimension."""

    base = white_acceleration_process_noise(dt_s, acceleration_std_mps2)
    return _embed_matrix(base, state_dim=state_dim, diagonal_for_extra=0.0)


def _embed_matrix(matrix: np.ndarray, *, state_dim: int, diagonal_for_extra: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape == (state_dim, state_dim):
        return matrix
    if state_dim < matrix.shape[0]:
        raise ValueError(
            f"cannot embed {matrix.shape[0]}D CV matrix into {state_dim}D record state"
        )
    out = np.eye(state_dim) * float(diagonal_for_extra)
    out[: matrix.shape[0], : matrix.shape[1]] = matrix
    return out
