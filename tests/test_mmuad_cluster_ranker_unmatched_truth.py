from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.cluster_ranker import label_cluster_features_against_truth


_TRUTH_LABEL_COLUMNS = (
    "good_cluster_2m",
    "good_cluster_5m",
    "good_cluster_10m",
    "good_cluster_20m",
    "good_cluster",
)


def test_unmatched_cluster_candidates_remain_unlabeled() -> None:
    features = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seq_without_truth"],
            "time_s": [0.0, 10.0, 0.0],
            "x_m": [1.0, 1.0, 1.0],
            "y_m": [2.0, 2.0, 2.0],
            "z_m": [3.0, 3.0, 3.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )

    labeled = label_cluster_features_against_truth(
        features,
        truth,
        max_truth_time_delta_s=0.5,
    )

    assert labeled["truth_matched"].tolist() == [True, False, False]
    for column in _TRUTH_LABEL_COLUMNS:
        assert bool(labeled.loc[0, column])
        assert pd.isna(labeled.loc[1, column])
        assert pd.isna(labeled.loc[2, column])
        assert str(labeled[column].dtype) == "boolean"
