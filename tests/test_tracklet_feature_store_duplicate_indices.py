from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.tracklet_feature_store import (
    build_counterfactual_association_dashboard,
)


def test_dashboard_preserves_oracle_best_with_duplicate_indices() -> None:
    features = pd.DataFrame(
        {
            "frame_key_type": ["frame_index", "frame_index"],
            "frame_key": ["7", "7"],
            "time_s": [1.0, 1.0],
            "track_id": [11, 22],
            "track_index": [0, 1],
            "oracle_error_m": [2.0, 9.0],
            "oracle_rank_in_frame": [1.0, 2.0],
            "chosen_by_selected_radar": [False, True],
        },
        index=[5, 5],
    )

    dashboard = build_counterfactual_association_dashboard(features)

    assert len(dashboard) == 1
    assert np.isclose(dashboard.loc[0, "best_candidate_error_m"], 2.0)
    assert dashboard.loc[0, "best_candidate_track_id"] == 11
    assert np.isclose(dashboard.loc[0, "selected_candidate_error_m"], 9.0)
    assert np.isclose(dashboard.loc[0, "selection_regret_m"], 7.0)
    assert dashboard.loc[0, "category"] == "wrong_candidate_selected"
