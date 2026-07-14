from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.factor_graph import coordinate_descent_association_and_smoothing


def test_coordinate_descent_preserves_frames_when_frame_index_is_incomplete() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0.0, np.nan, 2.0],
            "track_id": [1, 1, 1],
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    _, selected = coordinate_descent_association_and_smoothing(
        radar,
        iterations=1,
        candidate_gate_m=10.0,
    )

    assert selected["time_s"].tolist() == [0.0, 1.0, 2.0]
