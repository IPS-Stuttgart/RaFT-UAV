from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir


def test_diversity_treats_non_finite_scores_as_missing() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [1.0, 1.0],
            "track_id": ["infinite", "finite"],
            "x_m": [0.0, 0.1],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [float("inf"), 1.0],
        }
    )

    output = diversify_candidate_reservoir(
        rows,
        radius_m=1.0,
        max_candidates_per_frame=1,
    )

    assert output["track_id"].tolist() == ["finite"]
    assert output["candidate_reservoir_score"].tolist() == [1.0]


def test_diversity_skips_malformed_coordinate_rows() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [1.0, 1.0],
            "track_id": ["malformed", "valid"],
            "x_m": ["not-a-number", "5.0"],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [10.0, 1.0],
        }
    )

    output = diversify_candidate_reservoir(rows, radius_m=0.0)

    assert output["track_id"].tolist() == ["valid"]
    assert output["x_m"].tolist() == [5.0]
