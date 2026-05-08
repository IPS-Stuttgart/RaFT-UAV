import numpy as np

from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker
from raft_uav.baselines.kalman import TrackingMeasurement


def test_imm_tracker_exposes_cv_tracker_interface():
    covariance = np.diag([10.0, 10.0, 10.0])
    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=np.array([0.0, 0.0, 0.0]),
        initial_time_s=0.0,
    )

    diagnostics = tracker.update(
        TrackingMeasurement(1.0, np.array([1.0, 0.0, 0.0]), covariance, "radar")
    )

    assert diagnostics.accepted
    assert tracker.state.shape == (6,)
    assert tracker.covariance_matrix.shape == (6, 6)
    np.testing.assert_allclose(tracker.mode_probabilities.sum(), 1.0)
