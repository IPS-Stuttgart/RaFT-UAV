from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_trajectory_regularizer import regularize_track5_estimates


def test_duplicate_timestamps_do_not_create_artificial_acceleration() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 4,
            "time_s": [0.0, 1.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 1.0, 2.0],
            "state_y_m": [0.0] * 4,
            "state_z_m": [5.0] * 4,
        }
    )

    regularized, _ = regularize_track5_estimates(
        estimates,
        smoothness_weight=7200.0,
        huber_delta_m=25.0,
        iterations=1,
        observation_sigma_m=1.0,
    )

    assert regularized["time_s"].tolist() == [0.0, 1.0, 1.0, 2.0]
    expected = np.column_stack(
        [
            np.array([0.0, 1.0, 1.0, 2.0]),
            np.zeros(4),
            np.full(4, 5.0),
        ]
    )
    actual = regularized[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    np.testing.assert_allclose(actual, expected, atol=2.0e-5)
