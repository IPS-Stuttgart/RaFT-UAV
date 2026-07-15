from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker


_INVALID_SCALES = [
    np.nan,
    np.inf,
    -np.inf,
    -1.0,
    "not-a-scale",
    True,
    np.array([1.0]),
]


@pytest.mark.parametrize(
    "parameter",
    [
        "initial_position_std_m",
        "initial_velocity_std_mps",
        "acceleration_std_mps2",
    ],
)
@pytest.mark.parametrize("value", _INVALID_SCALES)
def test_tracker_rejects_invalid_uncertainty_scales(parameter: str, value: object) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{parameter} must be a finite nonnegative scalar",
    ):
        AsyncConstantVelocityKalmanTracker(
            np.zeros(3),
            0.0,
            **{parameter: value},
        )


def test_tracker_normalizes_valid_uncertainty_scales() -> None:
    tracker = AsyncConstantVelocityKalmanTracker(
        np.zeros(3),
        0.0,
        initial_position_std_m="0.0",
        initial_velocity_std_mps=np.array(2.5),
        acceleration_std_mps2=0,
    )

    np.testing.assert_allclose(np.diag(tracker.covariance_matrix), [0.0, 0.0, 0.0, 6.25, 6.25, 6.25])
    assert tracker.acceleration_std_mps2 == 0.0
