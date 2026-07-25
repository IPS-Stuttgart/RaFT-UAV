from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker


def _tracker() -> AsyncConstantVelocityKalmanTracker:
    return AsyncConstantVelocityKalmanTracker(
        initial_position=np.array([1.0, 2.0, 3.0]),
        initial_time_s=0.0,
    )


@pytest.mark.parametrize(
    "time_s",
    [np.nan, np.inf, -np.inf, True, 1.0 + 0.0j, np.array([1.0]), np.ma.masked],
)
def test_coast_rejects_invalid_timestamps_without_mutation(time_s: object) -> None:
    tracker = _tracker()
    state_before = tracker.state
    covariance_before = tracker.covariance_matrix

    with pytest.raises(ValueError, match="time_s must be a finite numeric timestamp"):
        tracker.coast_to(time_s)

    assert tracker.current_time_s == 0.0
    assert tracker._initial_update_pending is True
    np.testing.assert_array_equal(tracker.state, state_before)
    np.testing.assert_array_equal(tracker.covariance_matrix, covariance_before)


def test_backward_coast_does_not_consume_bootstrap_state() -> None:
    tracker = _tracker()

    with pytest.raises(
        ValueError,
        match="measurements must be processed in chronological order",
    ):
        tracker.coast_to(-1.0)

    assert tracker.current_time_s == 0.0
    assert tracker._initial_update_pending is True


def test_coast_accepts_zero_dimensional_timestamp() -> None:
    tracker = _tracker()

    tracker.coast_to(np.array(1.0))

    assert tracker.current_time_s == 1.0
    assert tracker._initial_update_pending is False
