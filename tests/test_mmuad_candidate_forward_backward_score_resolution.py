from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_forward_backward import (
    CandidateForwardBackwardConfig,
    attach_forward_backward_candidate_prior,
)


def _single_frame_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 0.0],
            "source": ["lidar", "lidar"],
            "track_id": ["primary", "fallback"],
            "candidate_branch": ["raw", "raw"],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "primary_score": [0.2, np.nan],
            "fallback_score": [0.1, 0.9],
            "predicted_sigma_m": [1.0, 1.0],
        }
    )


def test_first_order_forward_backward_resolves_fallback_scores_per_row() -> None:
    augmented = attach_forward_backward_candidate_prior(
        _single_frame_candidates(),
        config=CandidateForwardBackwardConfig(
            score_column="primary_score",
            fallback_score_columns=("fallback_score",),
            score_normalization="none",
            sigma_log_weight=0.0,
        ),
    ).rows.set_index("track_id")

    assert augmented.loc["primary", "candidate_forward_backward_raw_score"] == pytest.approx(0.2)
    assert augmented.loc["fallback", "candidate_forward_backward_raw_score"] == pytest.approx(0.9)
    assert augmented.loc["fallback", "candidate_forward_backward_score"] > augmented.loc[
        "primary", "candidate_forward_backward_score"
    ]


def _tied_candidates(order: tuple[str, ...]) -> pd.DataFrame:
    score = {"a": 1.0, "b": 1.0, "c": 0.0}
    x = {"a": 0.0, "b": 0.0, "c": 0.0}
    return pd.DataFrame(
        {
            "sequence_id": ["seq001"] * len(order),
            "time_s": [0.0] * len(order),
            "source": ["lidar"] * len(order),
            "track_id": list(order),
            "candidate_branch": ["raw"] * len(order),
            "x_m": [x[item] for item in order],
            "y_m": [0.0] * len(order),
            "z_m": [0.0] * len(order),
            "ranker_score": [score[item] for item in order],
            "predicted_sigma_m": [1.0] * len(order),
        }
    )


def _rank_posterior(order: tuple[str, ...]) -> pd.DataFrame:
    return attach_forward_backward_candidate_prior(
        _tied_candidates(order),
        config=CandidateForwardBackwardConfig(
            score_column="ranker_score",
            score_normalization="rank",
            sigma_log_weight=0.0,
        ),
    ).rows.set_index("track_id")


def test_first_order_rank_ties_are_permutation_invariant() -> None:
    forward = _rank_posterior(("a", "b", "c"))
    reversed_rows = _rank_posterior(("c", "b", "a"))

    assert forward.loc["a", "candidate_forward_backward_score"] == pytest.approx(
        forward.loc["b", "candidate_forward_backward_score"]
    )
    assert forward.loc["a", "candidate_forward_backward_rank"] == pytest.approx(1.5)
    assert forward.loc["b", "candidate_forward_backward_rank"] == pytest.approx(1.5)
    assert forward.loc["c", "candidate_forward_backward_rank"] == pytest.approx(3.0)
    for track_id in ("a", "b", "c"):
        assert forward.loc[track_id, "candidate_forward_backward_score"] == pytest.approx(
            reversed_rows.loc[track_id, "candidate_forward_backward_score"]
        )
