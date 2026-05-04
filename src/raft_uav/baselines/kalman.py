"""Asynchronous constant-velocity Kalman fusion baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pyrecest.filters import KalmanFilter


@dataclass(frozen=True)
class TrackingMeasurement:
    """Position-like measurement in a local East-North-Up frame."""

    time_s: float
    vector: np.ndarray
    covariance: np.ndarray
    source: str

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=float).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=float)
        if vector.size not in (2, 3):
            raise ValueError("measurement vector must have 2 or 3 elements")
        if covariance.shape != (vector.size, vector.size):
            raise ValueError("measurement covariance must match vector dimension")


def constant_velocity_matrix(dt_s: float) -> np.ndarray:
    """Return the 6D constant-velocity transition matrix for ENU position/velocity."""

    dt = float(dt_s)
    matrix = np.eye(6)
    matrix[0, 3] = dt
    matrix[1, 4] = dt
    matrix[2, 5] = dt
    return matrix


def white_acceleration_process_noise(dt_s: float, acceleration_std: float) -> np.ndarray:
    """Return continuous white-acceleration process noise discretized for 3D CV."""

    dt = float(dt_s)
    q = float(acceleration_std) ** 2
    block = q * np.array(
        [
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2],
        ]
    )
    covariance = np.zeros((6, 6))
    for pos_idx, vel_idx in ((0, 3), (1, 4), (2, 5)):
        covariance[np.ix_([pos_idx, vel_idx], [pos_idx, vel_idx])] = block
    return covariance


def measurement_matrix(measurement_dim: int) -> np.ndarray:
    """Return a matrix mapping the 6D state to 2D or 3D position observations."""

    if measurement_dim == 2:
        matrix = np.zeros((2, 6))
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        return matrix
    if measurement_dim == 3:
        matrix = np.zeros((3, 6))
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        matrix[2, 2] = 1.0
        return matrix
    raise ValueError("measurement_dim must be 2 or 3")


class AsyncConstantVelocityKalmanTracker:
    """Small wrapper around PyRecEst's KalmanFilter for asynchronous sensor fusion."""

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_time_s: float,
        initial_position_std_m: float = 50.0,
        initial_velocity_std_mps: float = 15.0,
        acceleration_std_mps2: float = 4.0,
    ) -> None:
        position = np.asarray(initial_position, dtype=float).reshape(-1)
        if position.size == 2:
            position = np.array([position[0], position[1], 0.0])
        if position.size != 3:
            raise ValueError("initial_position must contain 2 or 3 elements")

        mean = np.zeros(6)
        mean[:3] = position
        covariance = np.diag(
            [
                initial_position_std_m**2,
                initial_position_std_m**2,
                initial_position_std_m**2,
                initial_velocity_std_mps**2,
                initial_velocity_std_mps**2,
                initial_velocity_std_mps**2,
            ]
        )
        self.filter = KalmanFilter((mean, covariance))
        self.current_time_s = float(initial_time_s)
        self.acceleration_std_mps2 = float(acceleration_std_mps2)

    @property
    def state(self) -> np.ndarray:
        """Return the current posterior mean."""

        return np.asarray(self.filter.get_point_estimate(), dtype=float)

    def predict_to(self, time_s: float) -> None:
        """Predict to an absolute timestamp."""

        dt_s = float(time_s) - self.current_time_s
        if dt_s < -1e-9:
            raise ValueError("measurements must be processed in chronological order")
        if dt_s > 0.0:
            self.filter.predict_linear(
                constant_velocity_matrix(dt_s),
                white_acceleration_process_noise(dt_s, self.acceleration_std_mps2),
            )
            self.current_time_s = float(time_s)

    def update(self, measurement: TrackingMeasurement) -> None:
        """Predict to and update from one RF or radar measurement."""

        self.predict_to(measurement.time_s)
        vector = np.asarray(measurement.vector, dtype=float).reshape(-1)
        self.filter.update_linear(
            vector,
            measurement_matrix(vector.size),
            np.asarray(measurement.covariance, dtype=float),
        )


def run_async_cv_baseline(
    measurements: Iterable[TrackingMeasurement],
    acceleration_std_mps2: float = 4.0,
) -> list[dict[str, object]]:
    """Run the asynchronous CV Kalman baseline and return posterior state records."""

    ordered = sorted(measurements, key=lambda item: item.time_s)
    if not ordered:
        return []

    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=ordered[0].vector,
        initial_time_s=ordered[0].time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )

    records: list[dict[str, object]] = []
    for measurement in ordered:
        tracker.update(measurement)
        records.append(
            {
                "time_s": measurement.time_s,
                "source": measurement.source,
                "state": tracker.state.copy(),
            }
        )
    return records
