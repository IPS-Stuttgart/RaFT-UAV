from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_forward_backward import (
    CandidateForwardBackwardConfig,
    attach_forward_backward_candidate_prior,
)
from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    attach_pair_forward_backward_candidate_prior,
)


def _single_frame_candidates(invalid_sigma: object) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["invalid", "fallback"],
            "candidate_branch": ["raw", "raw"],
            "x_m": [1.0, 1.0],
            "y_m": [2.0, 2.0],
            "z_m": [3.0, 3.0],
            "ranker_score": [0.5, 0.5],
        }
    )
    rows["predicted_sigma_m"] = pd.Series(
        [invalid_sigma, 7.0],
        dtype=object,
    )
    return rows


_INVALID_SIGMAS = (
    0.0,
    -3.0,
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    np.bool_(False),
)


@pytest.mark.parametrize("invalid_sigma", _INVALID_SIGMAS)
def test_first_order_prior_falls_back_for_invalid_candidate_sigma(
    invalid_sigma: object,
) -> None:
    augmented = attach_forward_backward_candidate_prior(
        _single_frame_candidates(invalid_sigma),
        config=CandidateForwardBackwardConfig(
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            default_sigma_m=7.0,
            sigma_min_m=1.0,
            sigma_max_m=30.0,
            score_normalization="none",
        ),
    ).rows.set_index("track_id")

    assert augmented.loc["invalid", "candidate_forward_backward_sigma_m"] == pytest.approx(
        7.0
    )
    assert augmented.loc[
        "invalid", "candidate_forward_backward_score"
    ] == pytest.approx(0.5)
    assert augmented.loc[
        "fallback", "candidate_forward_backward_score"
    ] == pytest.approx(0.5)


@pytest.mark.parametrize("invalid_sigma", _INVALID_SIGMAS)
def test_pair_state_prior_falls_back_for_invalid_candidate_sigma(
    invalid_sigma: object,
) -> None:
    augmented = attach_pair_forward_backward_candidate_prior(
        _single_frame_candidates(invalid_sigma),
        config=CandidatePairForwardBackwardConfig(
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            default_sigma_m=7.0,
            sigma_min_m=1.0,
            sigma_max_m=30.0,
            score_normalization="none",
        ),
    ).rows.set_index("track_id")

    assert augmented.loc[
        "invalid", "candidate_pair_forward_backward_sigma_m"
    ] == pytest.approx(7.0)
    assert augmented.loc[
        "invalid", "candidate_pair_forward_backward_score"
    ] == pytest.approx(0.5)
    assert augmented.loc[
        "fallback", "candidate_pair_forward_backward_score"
    ] == pytest.approx(0.5)
