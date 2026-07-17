from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.factor_graph import coordinate_descent_association_and_smoothing


def test_coordinate_descent_uses_time_for_missing_frame_index() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [np.nan, np.nan, np.nan],
            "track_id": [1, 2, 3],
            "time_s": [0.0, 0.0, 1.0],
            "east_m": [100.0, 0.0, 1.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [0.1, 0.9, 0.8],
        }
    )

    trajectory, selected = coordinate_descent_association_and_smoothing(
        radar,
        iterations=0,
    )

    assert selected["track_id"].tolist() == [2, 3]
    np.testing.assert_allclose(trajectory["time_s"], [0.0, 1.0])
