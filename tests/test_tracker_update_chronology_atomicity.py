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
def test_backward_update_does_not_consume_bootstrap_state(tracker_class) -> None:
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
    stale = TrackingMeasurement(
        time_s=9.0,
        vector=np.array([9.0, 8.0, 7.0]),
        covariance=np.eye(3),
        source="rf",
    )
    state_before = tracker.state
    covariance_before = tracker.covariance_matrix

    with pytest.raises(
        ValueError,
        match="measurements must be processed in chronological order",
    ):
        tracker.update(stale)

    assert tracker.current_time_s == 10.0
    assert tracker._initial_update_pending is True
    np.testing.assert_array_equal(tracker.state, state_before)
    np.testing.assert_array_equal(tracker.covariance_matrix, covariance_before)

    diagnostics = tracker.update(bootstrap)

    assert diagnostics.update_action == "initialized"
    np.testing.assert_array_equal(tracker.covariance_matrix, covariance_before)
