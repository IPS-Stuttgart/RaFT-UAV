from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    run_candidate_mixture_map,
)


def test_nearest_target_fallback_keeps_all_candidates_at_timestamp() -> None:
    candidates = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "left-a",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "radar",
                "track_id": "left-b",
                "x_m": 10.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 2.0,
                "source": "lidar_360",
                "track_id": "right",
                "x_m": 2.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    )
    target_template = pd.DataFrame(
        {
            "Sequence": ["seqA"],
            "Timestamp": [1.0],
        }
    )

    result = run_candidate_mixture_map(
        candidates,
        target_template=target_template,
        config=CandidateMixtureMapConfig(
            top_k=0,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=0.0,
            target_time_tolerance_s=0.1,
            iterations=1,
        ),
    )

    assignments = result.assignments.loc[result.assignments["time_s"] == 1.0]
    assert set(assignments["track_id"].astype(str)) == {"left-a", "left-b"}
    assert len(assignments) == 2
