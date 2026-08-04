from __future__ import annotations

import pandas as pd

from raft_uav.research.runtime_modes import backward_repair_associations


def test_backward_repair_compares_all_candidates_with_same_time_key() -> None:
    selected = pd.DataFrame(
        {
            "track_id": [1, 1],
            "time_s": [0.0, 2.0],
            "east_m": [0.0, 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "track_id": [10, 20],
            "time_s": [1.0000000001, 1.0000000002],
            "east_m": [1.05, 1.0000000002],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    repaired = backward_repair_associations(
        selected,
        candidates,
        max_gap_s=3.0,
        max_repair_distance_m=0.1,
    )

    inserted = repaired.loc[repaired["association_repaired"].fillna(False)]
    assert inserted["track_id"].tolist() == [20]
    assert inserted["association_score"].iloc[0] < 1e-8
