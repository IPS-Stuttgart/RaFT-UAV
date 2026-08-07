from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.tracklet_feature_store import (
    build_counterfactual_association_dashboard,
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
    assert features.loc[30, "frame_key_type"] == "frame_index_time_s"
    assert features.loc[30, "frame_key"] == "7@1.0"
    assert features.loc[10, "candidate_count_in_frame"] == 1
    assert features.loc[20, "candidate_count_in_frame"] == 1
    assert bool(features.loc[20, "chosen_by_selected_radar"])


def test_reused_frame_index_is_separated_by_timestamp() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "frame_index": [7.0, 7.0],
            "track_index": [0, 0],
            "track_id": [5, 5],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.9, 0.8],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    features = build_tracklet_candidate_feature_store(
        radar=radar,
        truth=truth,
        selected_radar=radar.iloc[[0]].copy(),
        truth_time_gate_s=0.1,
    ).sort_values("time_s")
    dashboard = build_counterfactual_association_dashboard(features).sort_values("time_s")

    assert features["frame_key_type"].tolist() == [
        "frame_index_time_s",
        "frame_index_time_s",
    ]
    assert features["frame_key"].tolist() == ["7@0.0", "7@10.0"]
    assert features["candidate_count_in_frame"].astype(int).tolist() == [1, 1]
    assert features["chosen_by_selected_radar"].astype(bool).tolist() == [True, False]
    assert dashboard["time_s"].tolist() == [0.0, 10.0]
    assert dashboard["candidate_count"].astype(int).tolist() == [1, 1]
    assert dashboard["selected_present"].astype(bool).tolist() == [True, False]


def test_frame_index_only_selection_matches_one_physical_frame() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [2.0],
            "frame_index": [9.0],
            "track_index": [0],
            "track_id": [42],
            "east_m": [2.0],
            "north_m": [0.0],
            "up_m": [0.0],
            "cat_prob_uav": [0.9],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [2.0],
            "east_m": [2.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    selected = radar.drop(columns=["time_s"])

    features = build_tracklet_candidate_feature_store(
        radar=radar,
        truth=truth,
        selected_radar=selected,
        truth_time_gate_s=0.1,
    )

    assert features["chosen_by_selected_radar"].astype(bool).tolist() == [True]


def test_ambiguous_frame_index_only_selection_does_not_leak_across_time() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "frame_index": [7.0, 7.0],
            "track_index": [0, 0],
            "track_id": [5, 5],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.9, 0.8],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    selected = radar.iloc[[0]].drop(columns=["time_s"])

    features = build_tracklet_candidate_feature_store(
        radar=radar,
        truth=truth,
        selected_radar=selected,
        truth_time_gate_s=0.1,
    )

    assert features["chosen_by_selected_radar"].astype(bool).tolist() == [False, False]
