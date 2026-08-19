from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
)


def _tracker() -> AsyncConstantVelocityKalmanTracker:
    return AsyncConstantVelocityKalmanTracker(
        initial_position=np.array([1.0, 2.0, 3.0]),
        initial_time_s=10.0,
    )


def _bootstrap_measurement() -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=10.0,
        vector=np.array([1.0, 2.0, 3.0]),
        covariance=np.eye(3),
        source="radar",
    )


def test_predict_to_rejects_nonfinite_time() -> None:
    tracker = _tracker()

    with pytest.raises(ValueError, match="time_s must be a finite real scalar"):
        tracker.predict_to(np.nan)

    assert tracker.current_time_s == 10.0


def test_failed_coast_does_not_consume_bootstrap_measurement() -> None:
    tracker = _tracker()

    with pytest.raises(
        ValueError,
        match="measurements must be processed in chronological order",
    ):
        tracker.coast_to(9.0)

    diagnostics = tracker.update(_bootstrap_measurement())
    assert diagnostics.update_action == "initialized"


def test_failed_nonfinite_coast_does_not_consume_bootstrap_measurement() -> None:
    tracker = _tracker()

    with pytest.raises(ValueError, match="time_s must be a finite real scalar"):
        tracker.coast_to(np.nan)

    diagnostics = tracker.update(_bootstrap_measurement())
    assert diagnostics.update_action == "initialized"


def test_failed_early_update_does_not_consume_bootstrap_measurement() -> None:
    tracker = _tracker()
    early = TrackingMeasurement(
        time_s=9.0,
        vector=np.array([1.0, 2.0, 3.0]),
        covariance=np.eye(3),
        source="radar",
    )

    with pytest.raises(
        ValueError,
        match="measurements must be processed in chronological order",
    ):
        tracker.update(early)

    diagnostics = tracker.update(_bootstrap_measurement())
    assert diagnostics.update_action == "initialized"
