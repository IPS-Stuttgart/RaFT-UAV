import numpy as np

from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker
from raft_uav.baselines.kalman import TrackingMeasurement


def test_imm_bootstrap_detection_uses_absolute_timestamp_tolerance() -> None:
    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=np.array([10.0, 20.0, 30.0]),
        initial_time_s=1000.0,
        acceleration_std_mps2=0.0,
    )
    initial_covariance = tracker.covariance_matrix
    measurement = TrackingMeasurement(
        time_s=1000.005,
        vector=np.array([10.0, 20.0, 30.0]),
        covariance=np.eye(3),
        source="rf",
    )

    diagnostics = tracker.update(measurement)

    assert diagnostics.update_action != "initialized"
    assert diagnostics.accepted is True
    assert tracker.current_time_s == measurement.time_s
    assert tracker.covariance_matrix[0, 0] < initial_covariance[0, 0]
