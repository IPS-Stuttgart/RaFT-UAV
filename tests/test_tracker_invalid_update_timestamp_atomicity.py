from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker
from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
)


_TRACKER_CLASSES = (
    AsyncConstantVelocityKalmanTracker,
    AsyncInteractingMultipleModelTracker,
)


@pytest.mark.parametrize("tracker_class", _TRACKER_CLASSES)
@pytest.mark.parametrize("invalid_time_s", [np.nan, np.inf])
def test_invalid_update_timestamp_does_not_consume_bootstrap_state(
    tracker_class,
    invalid_time_s: float,
) -> None:
    bootstrap = TrackingMeasurement(
        time_s=10.0,
        vector=np.array([1.0, 2.0, 3.0]),
        covariance=np.eye(3),
        source="radar",
    )
    tracker = tracker_class(
        initial_position=bootstrap.vector,
        initial_time_s=bootstrap.time_s,
    )
    malformed = TrackingMeasurement(
        time_s=11.0,
        vector=np.array([9.0, 8.0, 7.0]),
        covariance=np.eye(3),
        source="rf",
    )
    object.__setattr__(malformed, "time_s", invalid_time_s)

    state_before = tracker.state
    covariance_before = tracker.covariance_matrix

    with pytest.raises(
        ValueError,
        match="measurement time_s must be a finite numeric timestamp",
    ):
        tracker.update(malformed)

    assert tracker.current_time_s == 10.0
    assert tracker._initial_update_pending is True
    np.testing.assert_array_equal(tracker.state, state_before)
    np.testing.assert_array_equal(tracker.covariance_matrix, covariance_before)

    diagnostics = tracker.update(bootstrap)

    assert diagnostics.update_action == "initialized"
    np.testing.assert_array_equal(tracker.covariance_matrix, covariance_before)
