from __future__ import annotations

import pandas as pd

from raft_uav.diagnostics.tracklet_feature_store import (
    build_counterfactual_association_dashboard,
    build_tracklet_candidate_feature_store,
)


def test_feature_store_uses_time_keys_when_frame_indices_are_incomplete() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "frame_index": [0.0, float("nan"), float("nan")],
            "track_index": [0, 1, 2],
            "track_id": [11, 12, 13],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [0.9, 0.9, 0.9],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    features = build_tracklet_candidate_feature_store(
        radar=radar,
        truth=truth,
        truth_time_gate_s=1.0,
    )
    dashboard = build_counterfactual_association_dashboard(features)

    assert features["frame_key_type"].eq("time_s").all()
    assert features["candidate_count_in_frame"].tolist() == [1, 1, 1]
    assert features["oracle_best_candidate"].tolist() == [True, True, True]
    assert dashboard["time_s"].tolist() == [0.0, 1.0, 2.0]
