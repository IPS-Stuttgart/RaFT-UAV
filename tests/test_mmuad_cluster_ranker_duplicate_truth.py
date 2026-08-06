from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.cluster_ranker import label_cluster_features_against_truth


def test_cluster_ranker_uses_final_same_time_truth_row() -> None:
    features = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": ["0", 0.0],
            "x_m": [100.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    labeled = label_cluster_features_against_truth(
        features,
        truth,
        good_threshold_m=5.0,
        max_truth_time_delta_s=0.0,
    )

    assert labeled.loc[0, "truth_matched"]
    assert labeled.loc[0, "truth_distance_3d_m"] == 0.0
    assert labeled.loc[0, "good_cluster"]
