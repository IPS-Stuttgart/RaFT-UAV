from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.runtime_modes import backward_repair_associations


def test_backward_repair_scopes_reused_frame_indices_by_sequence() -> None:
    selected = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB", "seqA", "seqB"],
            "frame_index": [0, 0, 2, 2],
            "track_id": [1, 2, 1, 2],
            "time_s": [0.0, 0.0, 2.0, 2.0],
            "east_m": [0.0, 100.0, 2.0, 102.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB", "seqA", "seqB", "seqA", "seqB"],
            "frame_index": [0, 0, 1, 1, 2, 2],
            "track_id": [1, 2, 10, 20, 1, 2],
            "time_s": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "east_m": [0.0, 100.0, 1.0, 101.0, 2.0, 102.0],
            "north_m": [0.0] * 6,
            "up_m": [0.0] * 6,
        }
    )

    repaired = backward_repair_associations(
        selected,
        candidates,
        max_gap_s=3.0,
        max_repair_distance_m=10.0,
    )

    middle = repaired.loc[repaired["frame_index"] == 1]
    assert middle["sequence_id"].tolist() == ["seqA", "seqB"]
    assert middle["track_id"].tolist() == [10, 20]
    assert middle["association_repaired"].tolist() == [True, True]


def test_backward_repair_uses_time_when_frame_index_is_incomplete() -> None:
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
        max_repair_distance_m=10.0,
    )

    assert repaired["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert repaired.loc[repaired["time_s"] == 1.0, "track_id"].tolist() == [99]
