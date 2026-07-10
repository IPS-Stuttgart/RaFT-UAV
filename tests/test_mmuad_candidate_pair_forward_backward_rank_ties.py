from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    attach_pair_forward_backward_candidate_prior,
)


def _candidates(order: tuple[str, ...]) -> pd.DataFrame:
    scores = {"tied-a": 1.0, "tied-b": 1.0, "lower": 0.0}
    x = {"tied-a": 0.0, "tied-b": 1.0, "lower": 2.0}
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * len(order),
            "time_s": [0.0] * len(order),
            "source": ["candidate"] * len(order),
            "track_id": list(order),
            "candidate_branch": ["raw"] * len(order),
            "x_m": [x[track_id] for track_id in order],
            "y_m": [0.0] * len(order),
            "z_m": [0.0] * len(order),
            "ranker_score": [scores[track_id] for track_id in order],
            "predicted_sigma_m": [1.0] * len(order),
        }
    )


def _posterior(order: tuple[str, ...]) -> pd.DataFrame:
    return attach_pair_forward_backward_candidate_prior(
        _candidates(order),
        config=CandidatePairForwardBackwardConfig(
            score_column="ranker_score",
            fallback_score_columns=(),
            score_normalization="rank",
            sigma_log_weight=0.0,
        ),
    ).rows.set_index("track_id").sort_index()


def test_pair_forward_backward_rank_ties_are_permutation_invariant() -> None:
    forward = _posterior(("tied-a", "tied-b", "lower"))
    reversed_rows = _posterior(("lower", "tied-b", "tied-a"))

    pd.testing.assert_series_equal(
        forward["candidate_pair_forward_backward_score"],
        reversed_rows["candidate_pair_forward_backward_score"],
    )
    pd.testing.assert_series_equal(
        forward["candidate_pair_forward_backward_rank"],
        reversed_rows["candidate_pair_forward_backward_rank"],
    )
    assert forward.loc["tied-a", "candidate_pair_forward_backward_score"] == pytest.approx(
        forward.loc["tied-b", "candidate_pair_forward_backward_score"]
    )
    assert forward.loc["tied-a", "candidate_pair_forward_backward_rank"] == pytest.approx(1.5)
    assert forward.loc["tied-b", "candidate_pair_forward_backward_rank"] == pytest.approx(1.5)
    assert forward.loc["lower", "candidate_pair_forward_backward_rank"] == pytest.approx(3.0)
    assert forward.loc["tied-a", "candidate_pair_forward_backward_score"] > forward.loc[
        "lower", "candidate_pair_forward_backward_score"
    ]
    assert forward["candidate_pair_forward_backward_score"].sum() == pytest.approx(1.0)
