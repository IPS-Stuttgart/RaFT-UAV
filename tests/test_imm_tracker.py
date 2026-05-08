import numpy as np

from raft_uav.baselines.imm import (
    AsyncInteractingMultipleModelTracker,
    run_async_imm_baseline,
)
from raft_uav.baselines.kalman import TrackingMeasurement


def test_imm_tracker_accepts_6d_initial_state() -> None:
    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        initial_time_s=0.0,
    )

    assert np.allclose(tracker.state, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert set(tracker.mode_probability_map) == set(tracker.mode_names)
    assert np.isclose(sum(tracker.mode_probability_map.values()), 1.0)
    assert tracker.most_likely_mode_name in tracker.mode_names


def test_run_async_imm_baseline_smoke() -> None:
    measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([0.0, 0.0]),
            covariance=np.diag([4.0, 4.0]),
            source="rf",
        ),
        TrackingMeasurement(
            time_s=1.0,
            vector=np.array([1.0, 0.0, 0.0]),
            covariance=np.diag([4.0, 4.0, 9.0]),
            source="radar",
        ),
    ]

    records = run_async_imm_baseline(measurements, acceleration_std_mps2=2.0)

    assert len(records) == 2
    assert records[-1]["state"].shape == (6,)
    assert records[-1]["covariance"].shape == (6, 6)
    assert "mode_probability_map" in records[-1]
    assert "most_likely_mode" in records[-1]
    assert np.isclose(sum(records[-1]["mode_probability_map"].values()), 1.0)
