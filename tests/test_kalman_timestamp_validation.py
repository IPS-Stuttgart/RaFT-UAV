import numpy as np
import pytest

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
)


_INVALID_TIMESTAMPS = [np.nan, np.inf, -np.inf, "not-a-time", True, np.array([0.0])]


@pytest.mark.parametrize("time_s", _INVALID_TIMESTAMPS)
def test_tracking_measurement_rejects_invalid_timestamp(time_s):
    covariance = np.eye(3)

    with pytest.raises(ValueError, match="measurement time_s must be a finite numeric timestamp"):
        TrackingMeasurement(time_s, np.zeros(3), covariance, "radar")


def test_tracking_measurement_preserves_numeric_string_timestamp():
    measurement = TrackingMeasurement("1.25", np.zeros(3), np.eye(3), "radar")

    assert measurement.time_s == 1.25


@pytest.mark.parametrize("time_s", _INVALID_TIMESTAMPS)
def test_tracker_rejects_invalid_initial_timestamp(time_s):
    with pytest.raises(ValueError, match="initial_time_s must be a finite numeric timestamp"):
        AsyncConstantVelocityKalmanTracker(np.zeros(3), time_s)


@pytest.mark.parametrize("time_s", _INVALID_TIMESTAMPS)
def test_tracker_rejects_invalid_prediction_timestamp(time_s):
    tracker = AsyncConstantVelocityKalmanTracker(np.zeros(3), 0.0)

    with pytest.raises(ValueError, match="time_s must be a finite numeric timestamp"):
        tracker.predict_to(time_s)
