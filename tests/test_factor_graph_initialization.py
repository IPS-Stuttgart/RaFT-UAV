from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.factor_graph import (
    LeastSquaresSmoothingConfig,
    smooth_position_trajectory,
)


def test_factor_graph_sorts_and_collapses_initial_timestamps() -> None:
    measurements = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [100.0, 100.0, 100.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )
    initial = pd.DataFrame(
        {
            "time_s": [2.0, 0.0, 1.0, 1.0],
            "east_m": [20.0, 0.0, 8.0, 12.0],
            "north_m": [2.0, 0.0, 0.5, 1.5],
            "up_m": [4.0, 0.0, 1.0, 3.0],
        }
    )

    result = smooth_position_trajectory(
        measurements,
        initial=initial,
        config=LeastSquaresSmoothingConfig(max_nfev=1),
    )

    np.testing.assert_allclose(result.estimates["time_s"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        result.estimates[["east_m", "north_m", "up_m"]],
        [[0.0, 0.0, 0.0], [10.0, 1.0, 2.0], [20.0, 2.0, 4.0]],
    )
