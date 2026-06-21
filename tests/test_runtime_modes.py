from __future__ import annotations

import pandas as pd

from raft_uav.research.runtime_modes import _row_key, backward_repair_associations


def test_row_key_falls_back_to_time_for_invalid_frame_index() -> None:
    row = pd.Series({"frame_index": "not-a-frame", "time_s": 1.2345678912})

    assert _row_key(row) == round(1.2345678912, 9)


def test_backward_repair_falls_back_to_time_for_invalid_selected_frame_index() -> None:
    selected = pd.DataFrame(
        [
            {
                "frame_index": "anchor-a",
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
            },
            {
                "frame_index": "anchor-b",
                "time_s": 2.0,
                "east_m": 20.0,
                "north_m": 0.0,
                "up_m": 0.0,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
            }
        ]
    )

    repaired = backward_repair_associations(
        selected,
        candidates,
        max_gap_s=3.0,
        max_repair_distance_m=1.0,
    )

    assert repaired["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert bool(repaired.loc[repaired["time_s"].eq(1.0), "association_repaired"].iloc[0])
