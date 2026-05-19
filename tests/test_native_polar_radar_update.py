"""Tests for native radar-polar EKF updates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    RadarPolarMeasurement,
    enu_position_to_radar_polar,
)
from raft_uav.baselines.radar_association import _radar_row_to_measurement


def test_polar_radar_update_moves_position_toward_measurement() -> None:
    target = np.array([20.0, 100.0, 10.0])
    prior = np.array([40.0, 80.0, -5.0])
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=prior,
        initial_time_s=0.0,
        initial_position_std_m=30.0,
        initial_velocity_std_mps=1.0,
    )

    measurement = RadarPolarMeasurement(
        time_s=1.0,
        vector=enu_position_to_radar_polar(target),
        covariance=np.diag(
            [
                4.0**2,
                np.deg2rad(1.0) ** 2,
                np.deg2rad(1.0) ** 2,
            ]
        ),
        source="radar",
    )

    before = float(np.linalg.norm(tracker.state[:3] - target))
    diagnostics = tracker.update(measurement)
    after = float(np.linalg.norm(tracker.state[:3] - target))

    assert diagnostics.accepted
    assert diagnostics.source == "radar"
    assert diagnostics.measurement_dim == 3
    assert after < before


def test_radar_row_to_polar_measurement_prefers_logged_range() -> None:
    row = pd.Series(
        {
            "time_s": 3.0,
            "east_m": 3.0,
            "north_m": 4.0,
            "up_m": 0.0,
            "range_m": 10.0,
        }
    )

    measurement = _radar_row_to_measurement(
        row,
        np.diag([25.0**2, 25.0**2, 35.0**2]),
        measurement_model="polar-ekf",
    )

    assert isinstance(measurement, RadarPolarMeasurement)
    assert np.isclose(measurement.vector[0], 10.0)
    assert np.isclose(measurement.vector[1], np.arctan2(3.0, 4.0))
    assert np.isclose(measurement.vector[2], 0.0)
