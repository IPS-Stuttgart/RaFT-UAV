from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    compute_candidate_responsibilities,
)
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    HypothesisGroupConfig,
    prepare_hypothesis_group_candidates,
)


def _rank_candidates(order: tuple[str, ...]) -> pd.DataFrame:
    scores = {"tied-a": 1.0, "tied-b": 1.0, "lower": 0.0}
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * len(order),
            "time_s": [0.0] * len(order),
            "source": ["candidate"] * len(order),
            "track_id": list(order),
            "candidate_branch": ["raw"] * len(order),
            "candidate_origin_row": [f"origin-{name}" for name in order],
            "x_m": [0.0] * len(order),
            "y_m": [0.0] * len(order),
            "z_m": [0.0] * len(order),
            "ranker_score": [scores[name] for name in order],
            "predicted_sigma_m": [1.0] * len(order),
        }
    )


def _rank_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        fallback_score_columns=(),
        score_normalization="rank",
        score_weight=1.0,
        sigma_log_weight=0.0,
        loss="squared",
    )


def _responsibilities(order: tuple[str, ...]) -> pd.Series:
    result = compute_candidate_responsibilities(
        _rank_candidates(order),
        np.zeros(3),
        config=_rank_config(),
    )
    return result.set_index("track_id")["mixture_responsibility"].sort_index()


def test_mixture_rank_ties_are_permutation_invariant() -> None:
    forward = _responsibilities(("tied-a", "tied-b", "lower"))
    reversed_rows = _responsibilities(("lower", "tied-b", "tied-a"))

    pd.testing.assert_series_equal(forward, reversed_rows)
    assert forward.loc["tied-a"] == pytest.approx(forward.loc["tied-b"])
    assert forward.loc["tied-a"] > forward.loc["lower"]


def test_mixture_nonfinite_primary_score_uses_finite_fallback_before_top_k() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "source": ["candidate", "candidate"],
            "track_id": ["infinite-primary", "finite-primary"],
            "candidate_branch": ["raw", "raw"],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "primary_score": [np.inf, 1.0],
            "fallback_score": [0.0, 0.0],
            "predicted_sigma_m": [1.0, 1.0],
        }
    )

    result = compute_candidate_responsibilities(
        candidates,
        np.zeros(3),
        config=CandidateMixtureMapConfig(
            top_k=1,
            score_column="primary_score",
            fallback_score_columns=("fallback_score",),
            score_normalization="none",
            sigma_log_weight=0.0,
        ),
    )

    assert result["track_id"].tolist() == ["finite-primary"]


def _grouped_scores(order: tuple[str, ...]) -> pd.Series:
    prepared, _, _ = prepare_hypothesis_group_candidates(
        _rank_candidates(order),
        mixture_config=_rank_config(),
        group_config=HypothesisGroupConfig(correction_strength=0.0),
    )
    return prepared.set_index("track_id")[
        "mixture_group_base_normalized_score"
    ].sort_index()


def test_grouped_mixture_rank_ties_are_permutation_invariant() -> None:
    forward = _grouped_scores(("tied-a", "tied-b", "lower"))
    reversed_rows = _grouped_scores(("lower", "tied-b", "tied-a"))

    pd.testing.assert_series_equal(forward, reversed_rows)
    assert forward.loc["tied-a"] == pytest.approx(0.75)
    assert forward.loc["tied-b"] == pytest.approx(0.75)
    assert forward.loc["lower"] == pytest.approx(0.0)


def test_grouped_mixture_resolves_nonfinite_fallback_per_row() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "source": ["candidate", "candidate"],
            "track_id": ["primary", "fallback"],
            "candidate_branch": ["raw", "raw"],
            "candidate_origin_row": ["origin-primary", "origin-fallback"],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "primary_score": [10.0, np.inf],
            "fallback_score": [0.0, 7.0],
            "predicted_sigma_m": [1.0, 1.0],
        }
    )

    prepared, _, _ = prepare_hypothesis_group_candidates(
        candidates,
        mixture_config=CandidateMixtureMapConfig(
            score_column="primary_score",
            fallback_score_columns=("fallback_score",),
            score_normalization="rank",
            sigma_log_weight=0.0,
        ),
        group_config=HypothesisGroupConfig(correction_strength=0.0),
    )
    normalized = prepared.set_index("track_id")[
        "mixture_group_base_normalized_score"
    ]

    assert normalized.loc["primary"] == pytest.approx(1.0)
    assert normalized.loc["fallback"] == pytest.approx(0.0)
