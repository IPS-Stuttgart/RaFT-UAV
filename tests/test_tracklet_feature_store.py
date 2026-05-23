from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.tracklet_feature_store import (
    build_counterfactual_association_dashboard,
    build_tracklet_candidate_feature_store,
    summarize_counterfactual_regret,
)


def test_feature_store_marks_selected_regret() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_index": [0, 1],
            "track_id": [1, 2],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.2, 0.9],
        }
    )
    truth = pd.DataFrame(
        {"time_s": [0.0], "east_m": [0.0], "north_m": [0.0], "up_m": [0.0]}
    )
    selected = radar.iloc[[1]].copy()

    features = build_tracklet_candidate_feature_store(
        radar=radar,
        truth=truth,
        selected_radar=selected,
        truth_time_gate_s=1.0,
    )
    dashboard = build_counterfactual_association_dashboard(features)
    summary = summarize_counterfactual_regret(dashboard)

    assert int(features["oracle_best_candidate"].sum()) == 1
    assert int(features["chosen_by_selected_radar"].sum()) == 1
    assert dashboard.loc[0, "category"] == "wrong_candidate_selected"
    assert np.isclose(dashboard.loc[0, "selection_regret_m"], 100.0)
    assert summary["category_wrong_candidate_selected_count"] == 1
