from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import interpolate_truth_positions


def test_truth_interpolation_does_not_round_nearby_large_timestamp_to_anchor() -> None:
    truth = pd.DataFrame(
        {
            "time_s": [1000.0, 1000.01],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [5.0, 5.0],
        }
    )

    positions, valid = interpolate_truth_positions(
        truth,
        [1000.005],
        max_time_delta_s=0.01,
    )

    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_allclose(positions[0], [5.0, 0.0, 5.0])


def test_truth_interpolation_does_not_bypass_time_gate_at_large_timestamps() -> None:
    truth = pd.DataFrame(
        {
            "time_s": [999.0, 1000.005],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [5.0, 5.0],
        }
    )

    positions, valid = interpolate_truth_positions(
        truth,
        [1000.0],
        max_time_delta_s=0.001,
    )

    np.testing.assert_array_equal(valid, [False])
    assert np.isnan(positions[0]).all()
