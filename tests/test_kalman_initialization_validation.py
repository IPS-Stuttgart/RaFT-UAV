import numpy as np
import pytest

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("initial_time_s", np.nan),
        ("initial_time_s", np.array([0.0])),
        ("initial_position_std_m", -1.0),
        ("initial_velocity_std_mps", np.inf),
        ("acceleration_std_mps2", -4.0),
    ],
)
def test_tracker_rejects_invalid_scalar_initialization(keyword, value):
    kwargs = {keyword: value}

    with pytest.raises(ValueError, match=keyword):
        AsyncConstantVelocityKalmanTracker(
            initial_position=np.array([0.0, 0.0, 0.0]),
            initial_time_s=0.0,
            **kwargs,
        )


@pytest.mark.parametrize(
    "initial_position",
    [
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0 + 1.0j, 0.0, 0.0]),
        np.ma.array([0.0, 1.0, 2.0], mask=[False, True, False]),
    ],
)
def test_tracker_rejects_invalid_initial_positions(initial_position):
    with pytest.raises(ValueError, match="initial_position"):
        AsyncConstantVelocityKalmanTracker(
            initial_position=initial_position,
            initial_time_s=0.0,
        )


def test_tracker_accepts_zero_uncertainty_and_process_noise_scales():
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=np.array([1.0, 2.0, 3.0]),
        initial_time_s=-1.0,
        initial_position_std_m=0.0,
        initial_velocity_std_mps=0.0,
        acceleration_std_mps2=0.0,
    )

    np.testing.assert_allclose(tracker.state[:3], [1.0, 2.0, 3.0])
    assert tracker.current_time_s == -1.0
    assert tracker.acceleration_std_mps2 == 0.0
