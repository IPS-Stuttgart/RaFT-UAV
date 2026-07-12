from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
)
from raft_uav.mmuad.candidate_pair_group_correction import (
    PairGroupMultiplicityConfig,
    attach_group_corrected_pair_prior,
    prepare_group_corrected_pair_candidates,
)


def _rank_candidates(order: tuple[str, ...]) -> pd.DataFrame:
    scores = {"tied-a": 1.0, "tied-b": 1.0, "lower": 0.0}
    positions = {"tied-a": 0.0, "tied-b": 1.0, "lower": 2.0}
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * len(order),
            "time_s": [0.0] * len(order),
            "source": ["candidate"] * len(order),
            "track_id": list(order),
            "candidate_origin_row": [f"origin-{name}" for name in order],
            "x_m": [positions[name] for name in order],
            "y_m": [0.0] * len(order),
            "z_m": [0.0] * len(order),
            "ranker_score": [scores[name] for name in order],
            "predicted_sigma_m": [1.0] * len(order),
        }
    )


def _rank_pair_config() -> CandidatePairForwardBackwardConfig:
    return CandidatePairForwardBackwardConfig(
        score_column="ranker_score",
        fallback_score_columns=(),
        score_normalization="rank",
        sigma_log_weight=0.0,
        output_score_column="pair_score",
    )


def _rank_result(order: tuple[str, ...]) -> pd.DataFrame:
    augmented, _, _ = attach_group_corrected_pair_prior(
        _rank_candidates(order),
        pair_config=_rank_pair_config(),
        group_config=PairGroupMultiplicityConfig(correction_strength=0.0),
    )
    return augmented.rows.set_index("track_id").sort_index()


def test_group_correction_rank_ties_are_permutation_invariant() -> None:
    forward = _rank_result(("tied-a", "tied-b", "lower"))
    reversed_rows = _rank_result(("lower", "tied-b", "tied-a"))

    for column in (
        "candidate_pair_group_base_normalized_score",
        "pair_score",
        "candidate_pair_forward_backward_rank",
    ):
        pd.testing.assert_series_equal(forward[column], reversed_rows[column])

    assert forward.loc[
        "tied-a", "candidate_pair_group_base_normalized_score"
    ] == pytest.approx(0.75)
    assert forward.loc[
        "tied-b", "candidate_pair_group_base_normalized_score"
    ] == pytest.approx(0.75)
    assert forward.loc["tied-a", "pair_score"] == pytest.approx(
        forward.loc["tied-b", "pair_score"]
    )


def test_group_correction_resolves_score_fallbacks_per_candidate_row() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "source": ["candidate", "candidate"],
            "track_id": ["primary", "fallback"],
            "candidate_origin_row": ["origin-primary", "origin-fallback"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "primary_score": [10.0, None],
            "fallback_score": [0.0, 7.0],
            "predicted_sigma_m": [1.0, 1.0],
        }
    )
    prepared, _, _ = prepare_group_corrected_pair_candidates(
        candidates,
        pair_config=CandidatePairForwardBackwardConfig(
            score_column="primary_score",
            fallback_score_columns=("fallback_score",),
            score_normalization="rank",
            sigma_log_weight=0.0,
        ),
        group_config=PairGroupMultiplicityConfig(correction_strength=0.0),
    )
    normalized = prepared.set_index("track_id")[
        "candidate_pair_group_base_normalized_score"
    ]

    assert normalized.loc["primary"] == pytest.approx(1.0)
    assert normalized.loc["fallback"] == pytest.approx(0.0)
