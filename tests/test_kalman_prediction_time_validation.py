from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker


def _tracker() -> AsyncConstantVelocityKalmanTracker:
    return AsyncConstantVelocityKalmanTracker(
        initial_position=np.array([1.0, 2.0, 3.0]),
        initial_time_s=0.0,
    )


@pytest.mark.parametrize("method_name", ["predict_to", "coast_to"])
@pytest.mark.parametrize(
    "time_s",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
    ],
)
def test_tracker_rejects_malformed_prediction_timestamps_without_mutation(
    method_name: str,
    time_s: object,
) -> None:
    tracker = _tracker()
    state_before = tracker.state
    covariance_before = tracker.covariance_matrix
    prior_state_before = tracker.last_prior_state
    prior_covariance_before = tracker.last_prior_covariance_matrix

    with pytest.raises(ValueError, match="time_s must be a finite real scalar"):
        getattr(tracker, method_name)(time_s)

    assert tracker.current_time_s == 0.0
    assert tracker._initial_update_pending is True
    np.testing.assert_array_equal(tracker.state, state_before)
    np.testing.assert_array_equal(tracker.covariance_matrix, covariance_before)
    np.testing.assert_array_equal(tracker.last_prior_state, prior_state_before)
    np.testing.assert_array_equal(
        tracker.last_prior_covariance_matrix,
        prior_covariance_before,
    )


def test_failed_backward_coast_does_not_consume_bootstrap_state() -> None:
    tracker = _tracker()

    with pytest.raises(
        ValueError,
        match="measurements must be processed in chronological order",
    ):
        tracker.coast_to(-1.0)

    assert tracker.current_time_s == 0.0
    assert tracker._initial_update_pending is True


def test_tracker_accepts_zero_dimensional_prediction_timestamps() -> None:
    tracker = _tracker()

    tracker.predict_to(np.array(1.0))
    tracker.coast_to(np.array(2.0))

    assert tracker.current_time_s == 2.0
    assert tracker._initial_update_pending is False
    np.testing.assert_array_equal(tracker.last_prior_state, tracker.state)
    np.testing.assert_array_equal(
        tracker.last_prior_covariance_matrix,
        tracker.covariance_matrix,
    )
