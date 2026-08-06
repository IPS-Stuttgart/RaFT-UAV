from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.tracklet_feature_store import (
    build_tracklet_candidate_feature_store,
)


def test_partial_frame_indices_fall_back_to_each_row_timestamp() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 1.0],
            "frame_index": [np.nan, np.nan, 7.0],
            "track_index": [0, 0, 1],
            "track_id": [10, 20, 30],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [0.9, 0.8, 0.7],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    selected = radar.iloc[[1]].copy()

    features = build_tracklet_candidate_feature_store(
        radar=radar,
        truth=truth,
        selected_radar=selected,
        truth_time_gate_s=1.0,
    ).set_index("track_id")

    assert features.loc[10, "frame_key_type"] == "time_s"
    assert features.loc[10, "frame_key"] == "0.0"
    assert features.loc[20, "frame_key_type"] == "time_s"
    assert features.loc[20, "frame_key"] == "1.0"
    assert features.loc[30, "frame_key_type"] == "frame_index"
    assert features.loc[30, "frame_key"] == "7"
    assert features.loc[10, "candidate_count_in_frame"] == 1
    assert features.loc[20, "candidate_count_in_frame"] == 1
    assert bool(features.loc[20, "chosen_by_selected_radar"])
