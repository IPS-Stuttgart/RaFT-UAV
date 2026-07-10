from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    attach_pair_forward_backward_candidate_prior,
)


def test_pair_forward_backward_resolves_fallback_scores_per_row() -> None:
    candidates = pd.DataFrame(
        [
            {
                "sequence_id": "seq",
                "time_s": 0.0,
                "source": "lidar",
                "track_id": "primary",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "primary_score": 0.9,
                "fallback_score": 0.1,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seq",
                "time_s": 0.0,
                "source": "radar",
                "track_id": "fallback",
                "candidate_branch": "translated",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "primary_score": None,
                "fallback_score": 0.8,
                "predicted_sigma_m": 1.0,
            },
        ]
    )

    augmented = attach_pair_forward_backward_candidate_prior(
        candidates,
        config=CandidatePairForwardBackwardConfig(
            score_column="primary_score",
            fallback_score_columns=("fallback_score",),
            score_normalization="none",
            sigma_log_weight=0.0,
        ),
    ).rows.set_index("track_id")

    assert augmented.loc["primary", "candidate_pair_forward_backward_raw_score"] == pytest.approx(
        0.9
    )
    assert augmented.loc[
        "fallback", "candidate_pair_forward_backward_raw_score"
    ] == pytest.approx(0.8)
    assert augmented.loc["primary", "candidate_pair_forward_backward_score"] > augmented.loc[
        "fallback", "candidate_pair_forward_backward_score"
    ]
    assert augmented["candidate_pair_forward_backward_score"].sum() == pytest.approx(1.0)
