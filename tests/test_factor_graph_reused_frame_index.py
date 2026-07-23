from __future__ import annotations

import pandas as pd

from raft_uav.research.factor_graph import coordinate_descent_association_and_smoothing


def test_coordinate_descent_keeps_reused_frame_indices_separate() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, 0, 0, 0],
            "track_id": [1, 2, 3, 4],
            "time_s": [0.0, 0.0, 10.0, 10.0],
            "east_m": [0.0, 100.0, 10.0, 110.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
            "cat_prob_uav": [0.9, 0.1, 0.8, 0.2],
        }
    )

    trajectory, selected = coordinate_descent_association_and_smoothing(
        radar,
        iterations=0,
    )

    assert selected["track_id"].tolist() == [1, 3]
    assert trajectory["time_s"].tolist() == [0.0, 10.0]
