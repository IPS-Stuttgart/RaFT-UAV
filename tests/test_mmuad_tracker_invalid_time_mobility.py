from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.tracker import TrackerConfig, select_tracklet_path


def test_invalid_timestamp_candidate_does_not_change_mobility_ranking() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq", "seq", "seq"],
            "time_s": [0.0, 0.0, np.nan, 1.0],
            "source": ["candidate", "candidate", "candidate", "candidate"],
            "x_m": [0.0, 10.0, 0.0, 1.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "confidence": [0.9, 0.8, 0.1, 0.9],
        }
    )

    selected = select_tracklet_path(
        candidates,
        config=TrackerConfig(selection_mobility_radius_m=0.5),
    )

    assert selected["time_s"].tolist() == [0.0, 1.0]
    assert selected["x_m"].tolist() == [0.0, 1.0]
