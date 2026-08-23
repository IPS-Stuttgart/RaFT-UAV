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
def test_equal_independent_initial_measurement_is_assimilated(tracker_class):
    tracker = tracker_class(
        initial_position=np.zeros(3),
        initial_time_s=0.0,
        initial_position_std_m=50.0,
    )
    measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.zeros(3),
        covariance=np.eye(3),
        source="radar",
    )
    prior_covariance = tracker.covariance_matrix

    diagnostics = tracker.update(measurement)

    assert diagnostics.update_action != "initialized"
    assert np.all(
        np.diag(tracker.covariance_matrix)[:3]
        < np.diag(prior_covariance)[:3]
    )


@pytest.mark.parametrize("tracker_class", _TRACKER_CLASSES)
def test_measurement_vector_used_for_initialization_is_not_reassimilated(tracker_class):
    measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.zeros(3),
        covariance=np.eye(3),
        source="radar",
    )
    tracker = tracker_class(
        initial_position=measurement.vector,
        initial_time_s=measurement.time_s,
        initial_position_std_m=50.0,
    )
    prior_covariance = tracker.covariance_matrix

    diagnostics = tracker.update(measurement)

    assert diagnostics.update_action == "initialized"
    np.testing.assert_allclose(tracker.covariance_matrix, prior_covariance)
