from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.factor_graph import coordinate_descent_association_and_smoothing


def test_coordinate_descent_preserves_valid_frames_with_partial_indices() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0.0, 1.0, np.nan],
            "track_id": [1, 2, 3],
            "time_s": [0.0, 0.0, 1.0],
            "east_m": [0.0, 100.0, 1.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [0.9, 0.8, 0.7],
        }
    )

    trajectory, selected = coordinate_descent_association_and_smoothing(
        radar,
        iterations=0,
    )

    assert selected["track_id"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(trajectory["time_s"], [0.0, 1.0])
