from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_class_conditioned_anchor_quantile import (
    ClassConditionedAnchorReliabilityConfig,
    add_class_conditioned_anchor_quantile_selection_utility,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["lidar_360"],
            "track_id": ["candidate-0"],
            "origin_row": ["origin-0"],
            "candidate_branch": ["raw"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
            "ranker_score": [0.0],
            "predicted_sigma_m": [1.0],
        }
    )


def _anchor(sequence_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": [sequence_id],
            "Timestamp": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )


def _class_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "class_prob_0": [1.0],
            "class_prob_1": [0.0],
            "class_prob_2": [0.0],
            "class_prob_3": [0.0],
        }
    )


def _mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        fallback_score_columns=(),
        sigma_column="predicted_sigma_m",
        score_normalization="none",
        score_weight=0.0,
        temperature=1.0,
        sigma_log_weight=0.0,
        smoothness_weight=0.0,
        iterations=1,
    )


def _class_reliability() -> dict[str, dict[str, float]]:
    return {
        "active": {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0},
        "ignored": {"0": 0.0, "1": 1.0, "2": 1.0, "3": 1.0},
    }


def _add_utility(*, missing_anchor_policy: str) -> pd.DataFrame:
    scored, _, _ = add_class_conditioned_anchor_quantile_selection_utility(
        _candidates(),
        {
            "active": _anchor("seqB"),
            "ignored": _anchor("seqA"),
        },
        _class_probabilities(),
        anchor_reliability={"active": 1.0, "ignored": 1.0},
        anchor_class_reliability=_class_reliability(),
        mixture_config=_mixture_config(),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=1.0,
            anchor_scale_m=1.0,
            anchor_huber_delta=1.0,
            anchor_cost_cap=4.0,
            anchor_time_tolerance_s=0.01,
            missing_anchor_policy=missing_anchor_policy,
        ),
        reliability_config=ClassConditionedAnchorReliabilityConfig(
            cost_quantile=0.5,
            conditioning_strength=1.0,
        ),
    )
    return scored


def test_zero_effective_weight_anchor_does_not_satisfy_error_policy() -> None:
    with pytest.raises(
        ValueError,
        match="positive class-conditioned-reliability anchor",
    ):
        _add_utility(missing_anchor_policy="error")


def test_zero_effective_weight_only_support_remains_neutral_when_requested() -> None:
    scored = _add_utility(missing_anchor_policy="neutral")

    assert scored[
        "mixture_class_conditioned_anchor_quantile_matched_weight"
    ].tolist() == [0.0]
    assert scored[
        "mixture_class_conditioned_anchor_quantile_aggregate_cost"
    ].tolist() == [0.0]
