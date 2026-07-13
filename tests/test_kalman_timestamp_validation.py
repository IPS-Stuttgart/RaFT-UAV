from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
)


@pytest.mark.parametrize("time_s", [np.nan, np.inf, -np.inf])
def test_tracking_measurement_rejects_nonfinite_timestamp(time_s: float) -> None:
    with pytest.raises(ValueError, match="measurement time_s must be finite"):
        TrackingMeasurement(
            time_s=time_s,
            vector=np.zeros(3),
            covariance=np.eye(3),
            source="radar",
        )


@pytest.mark.parametrize("time_s", [np.nan, np.inf, -np.inf])
def test_tracker_rejects_nonfinite_initial_timestamp(time_s: float) -> None:
    with pytest.raises(ValueError, match="initial_time_s must be finite"):
        AsyncConstantVelocityKalmanTracker(
            initial_position=np.zeros(3),
            initial_time_s=time_s,
        )


@pytest.mark.parametrize("time_s", [np.nan, np.inf, -np.inf])
def test_tracker_rejects_nonfinite_prediction_timestamp(time_s: float) -> None:
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=np.zeros(3),
        initial_time_s=0.0,
    )

    with pytest.raises(ValueError, match="time_s must be finite"):
        tracker.predict_to(time_s)

    assert tracker.current_time_s == 0.0
