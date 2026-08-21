from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker
from raft_uav.baselines.kalman import TrackingMeasurement


@pytest.mark.parametrize(
    ("rejected_time_s", "error_match"),
    [
        (9.0, "measurements must be processed in chronological order"),
        (np.nan, "time_s must be a finite numeric timestamp"),
        (np.inf, "time_s must be a finite numeric timestamp"),
    ],
)
def test_rejected_imm_coast_does_not_consume_bootstrap_state(
    rejected_time_s: float,
    error_match: str,
) -> None:
    bootstrap = TrackingMeasurement(
        time_s=10.0,
        vector=np.array([1.0, 2.0, 3.0]),
        covariance=np.eye(3),
        source="radar",
    )
    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=bootstrap.vector,
        initial_time_s=bootstrap.time_s,
    )
    state_before = tracker.state
    covariance_before = tracker.covariance_matrix
    mode_probabilities_before = tracker.mode_probabilities

    with pytest.raises(ValueError, match=error_match):
        tracker.coast_to(rejected_time_s)

    assert tracker.current_time_s == 10.0
    assert tracker._initial_update_pending is True
    np.testing.assert_array_equal(tracker.state, state_before)
    np.testing.assert_array_equal(tracker.covariance_matrix, covariance_before)
    np.testing.assert_array_equal(
        tracker.mode_probabilities,
        mode_probabilities_before,
    )

    diagnostics = tracker.update(bootstrap)

    assert diagnostics.update_action == "initialized"
    np.testing.assert_array_equal(tracker.state, state_before)
    np.testing.assert_array_equal(tracker.covariance_matrix, covariance_before)
