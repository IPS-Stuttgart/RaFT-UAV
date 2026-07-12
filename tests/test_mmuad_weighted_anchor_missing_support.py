from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_anchor_quantile import (
    add_weighted_quantile_multi_anchor_conditioned_selection_utility,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_multi_anchor_mass_topk import (
    add_weighted_multi_anchor_conditioned_selection_utility,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig

AddUtility = Callable[..., tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]


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


@pytest.mark.parametrize(
    "add_utility",
    [
        add_weighted_multi_anchor_conditioned_selection_utility,
        add_weighted_quantile_multi_anchor_conditioned_selection_utility,
    ],
    ids=["weighted", "weighted-quantile"],
)
def test_zero_weight_anchor_does_not_satisfy_missing_support_error(
    add_utility: AddUtility,
) -> None:
    with pytest.raises(ValueError, match="positive-reliability anchor"):
        add_utility(
            _candidates(),
            {
                "positive": _anchor("seqB"),
                "ignored": _anchor("seqA"),
            },
            anchor_reliability={"positive": 1.0, "ignored": 0.0},
            mixture_config=_mixture_config(),
            anchor_config=AnchorConditioningConfig(
                anchor_selection_weight=1.0,
                anchor_scale_m=1.0,
                anchor_huber_delta=1.0,
                anchor_cost_cap=4.0,
                anchor_time_tolerance_s=0.01,
                missing_anchor_policy="error",
            ),
        )


@pytest.mark.parametrize(
    ("add_utility", "matched_weight_column"),
    [
        (
            add_weighted_multi_anchor_conditioned_selection_utility,
            "mixture_weighted_multi_anchor_matched_weight",
        ),
        (
            add_weighted_quantile_multi_anchor_conditioned_selection_utility,
            "mixture_weighted_quantile_multi_anchor_matched_weight",
        ),
    ],
    ids=["weighted", "weighted-quantile"],
)
def test_zero_weight_only_support_remains_neutral_when_requested(
    add_utility: AddUtility,
    matched_weight_column: str,
) -> None:
    scored, _, _ = add_utility(
        _candidates(),
        {
            "positive": _anchor("seqB"),
            "ignored": _anchor("seqA"),
        },
        anchor_reliability={"positive": 1.0, "ignored": 0.0},
        mixture_config=_mixture_config(),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=1.0,
            anchor_scale_m=1.0,
            anchor_huber_delta=1.0,
            anchor_cost_cap=4.0,
            anchor_time_tolerance_s=0.01,
            missing_anchor_policy="neutral",
        ),
    )

    assert scored[matched_weight_column].tolist() == [0.0]
