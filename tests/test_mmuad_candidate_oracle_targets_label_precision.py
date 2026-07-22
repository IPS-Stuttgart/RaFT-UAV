from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_oracle_targets import CandidateOracleTargetConfig
from raft_uav.mmuad.candidate_oracle_targets import build_candidate_oracle_targets


def test_high_precision_oracle_target_threshold_labels_remain_distinct() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["lidar_360", "livox_avia"],
            "track_id": ["oracle", "boundary"],
            "x_m": [0.0, 0.80000015],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.5, 0.4],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    target_rows, _, _ = build_candidate_oracle_targets(
        candidates,
        truth,
        config=CandidateOracleTargetConfig(
            soft_tau_m=(0.8000001, 0.8000002),
            good_thresholds_m=(0.8000001, 0.8000002),
        ),
    )

    lower_good = "candidate_good_le_0p8000001_m"
    upper_good = "candidate_good_le_0p8000002_m"
    lower_soft = "soft_oracle_weight_tau_0p8000001_m"
    upper_soft = "soft_oracle_weight_tau_0p8000002_m"
    assert {lower_good, upper_good, lower_soft, upper_soft}.issubset(target_rows.columns)

    boundary = target_rows.loc[target_rows["track_id"] == "boundary"].iloc[0]
    assert not bool(boundary[lower_good])
    assert bool(boundary[upper_good])
    assert float(boundary[lower_soft]) != float(boundary[upper_soft])
