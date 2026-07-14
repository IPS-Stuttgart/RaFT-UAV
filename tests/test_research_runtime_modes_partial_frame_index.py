from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.runtime_modes import backward_repair_associations


def test_backward_repair_preserves_candidate_with_missing_frame_index() -> None:
    selected = pd.DataFrame(
        {
            "frame_index": [0.0, 2.0],
            "track_id": [1, 1],
            "time_s": [0.0, 2.0],
            "east_m": [0.0, 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "frame_index": [0.0, np.nan, 2.0],
            "track_id": [1, 99, 1],
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    repaired = backward_repair_associations(
        selected,
        candidates,
        max_gap_s=3.0,
        max_repair_distance_m=0.1,
    )

    assert repaired["time_s"].tolist() == [0.0, 1.0, 2.0]
    middle = repaired.loc[repaired["time_s"] == 1.0].iloc[0]
    assert int(middle["track_id"]) == 99
    assert bool(middle["association_repaired"])
    assert float(middle["association_score"]) == 0.0
