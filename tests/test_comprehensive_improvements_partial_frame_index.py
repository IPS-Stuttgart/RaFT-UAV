from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.comprehensive_improvements import (
    _radar_frame_groups,
    candidate_recall_regret_table,
)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def test_candidate_diagnostics_keep_rows_with_partial_frame_indices() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "frame_index": [7.0, np.nan],
            "track_id": [1, 2],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    table = candidate_recall_regret_table(
        radar,
        _truth(),
        truth_gate_m=1.0,
        truth_time_gate_s=0.1,
    )

    assert table["time_s"].tolist() == [0.0, 1.0]
    assert table["frame_key_type"].tolist() == ["frame_index", "time_s"]
    assert table["candidate_rows"].tolist() == [1, 1]
    assert table["candidate_available"].astype(bool).tolist() == [True, True]


def test_serialized_missing_frame_indices_do_not_merge_distinct_times() -> None:
    radar = pd.DataFrame(
        {
            "time_s": ["10", "2"],
            "frame_index": ["<NA>", "<NA>"],
            "track_id": ["10", "2"],
            "east_m": [10.0, 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    groups = _radar_frame_groups(radar)

    assert len(groups) == 2
    assert [float(group["time_s"].iloc[0]) for group in groups] == [2.0, 10.0]
