from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import interpolate_truth_positions


def test_interpolation_sorts_numeric_string_timestamps_numerically() -> None:
    truth = pd.DataFrame(
        {
            "time_s": ["10", "0", "2"],
            "east_m": [100.0, 0.0, 4.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    positions, valid = interpolate_truth_positions(
        truth,
        [5.0],
        max_time_delta_s=5.0,
    )

    assert valid.tolist() == [True]
    np.testing.assert_allclose(positions[0], [40.0, 0.0, 0.0])
