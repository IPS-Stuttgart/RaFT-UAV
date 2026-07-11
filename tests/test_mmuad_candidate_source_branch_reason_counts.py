from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_source_branch_reservoir import (
    build_source_branch_reservoir,
)


def test_source_branch_quota_recomputes_reason_counts_before_final_cap() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0] * 4,
            "source": ["lidar_360", "lidar_360", "livox_avia", "livox_avia"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "track_id": [
                "lidar-raw",
                "lidar-translated",
                "livox-raw",
                "livox-translated",
            ],
            "x_m": [0.0, 1.0, 2.0, 3.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "ranker_score": [0.90, 0.10, 0.80, 0.70],
        }
    )
    bonus = 0.25
    reservoir = build_source_branch_reservoir(
        candidates,
        reservoir_config=ReservoirConfig(
            global_top_n=1,
            per_source_top_n=1,
            per_branch_top_n=1,
            max_candidates_per_frame=8,
            score_column="ranker_score",
            fallback_score_column="confidence",
            cap_reason_bonus=bonus,
        ),
        per_source_branch_top_n=1,
    ).rows.set_index("track_id")

    expected_counts = {
        "lidar-raw": 4,
        "lidar-translated": 1,
        "livox-raw": 2,
        "livox-translated": 2,
    }
    assert reservoir["candidate_reservoir_reason_count"].astype(int).to_dict() == expected_counts

    expected_cap_score = reservoir["candidate_reservoir_score"] + (
        bonus * reservoir["candidate_reservoir_reason_count"]
    )
    np.testing.assert_allclose(
        reservoir["candidate_reservoir_cap_score"].to_numpy(float),
        expected_cap_score.to_numpy(float),
    )
