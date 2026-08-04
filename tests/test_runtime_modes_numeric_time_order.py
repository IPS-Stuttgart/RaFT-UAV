from __future__ import annotations

import pandas as pd

from raft_uav.research.runtime_modes import backward_repair_associations


def test_backward_repair_sorts_numeric_string_timestamps_chronologically() -> None:
    selected = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq", "seq"],
            "track_id": [1, 1, 1],
            "time_s": ["1", "2", "10"],
            "east_m": [1.0, 2.0, 10.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "track_id": [99],
            "time_s": ["6"],
            "east_m": [6.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    repaired = backward_repair_associations(
        selected,
        candidates,
        max_gap_s=8.5,
        max_repair_distance_m=0.0,
    )

    numeric_times = pd.to_numeric(repaired["time_s"]).tolist()
    assert numeric_times == [1.0, 2.0, 6.0, 10.0]
    middle = repaired.loc[pd.to_numeric(repaired["time_s"]) == 6.0]
    assert middle["track_id"].tolist() == [99]
    assert middle["association_repaired"].tolist() == [True]
