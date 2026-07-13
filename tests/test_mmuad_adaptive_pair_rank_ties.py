from __future__ import annotations

from importlib import import_module

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
)


_VARIANTS = (
    pytest.param(
        "raft_uav.mmuad.candidate_pair_forward_backward_adaptive",
        "attach_entropy_adaptive_pair_prior",
        "EntropyAdaptivePairBlendConfig",
        "candidate_pair_forward_backward_local_posterior",
        "candidate_pair_forward_backward_adaptive_rank",
        id="entropy-adaptive",
    ),
    pytest.param(
        "raft_uav.mmuad.candidate_pair_forward_backward_agreement",
        "attach_agreement_adaptive_pair_prior",
        "AgreementAdaptivePairBlendConfig",
        "candidate_pair_forward_backward_local_posterior",
        "candidate_pair_forward_backward_agreement_rank",
        id="agreement",
    ),
    pytest.param(
        "raft_uav.mmuad.candidate_pair_forward_backward_agreement_adaptive",
        "attach_agreement_adaptive_pair_prior",
        "AgreementAdaptivePairBlendConfig",
        "candidate_pair_forward_backward_agreement_local_posterior",
        "candidate_pair_forward_backward_agreement_adaptive_rank",
        id="agreement-adaptive",
    ),
)


def _candidates(order: tuple[str, ...]) -> pd.DataFrame:
    scores = {"tied-a": 1.0, "tied-b": 1.0, "lower": 0.0}
    positions = {"tied-a": 0.0, "tied-b": 1.0, "lower": 2.0}
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * len(order),
            "time_s": [0.0] * len(order),
            "source": ["candidate"] * len(order),
            "track_id": list(order),
            "candidate_branch": ["raw"] * len(order),
            "x_m": [positions[name] for name in order],
            "y_m": [0.0] * len(order),
            "z_m": [0.0] * len(order),
            "ranker_score": [scores[name] for name in order],
            "predicted_sigma_m": [1.0] * len(order),
        }
    )


def _run_variant(
    module_name: str,
    attach_name: str,
    blend_config_name: str,
    order: tuple[str, ...],
) -> tuple[pd.DataFrame, str]:
    module = import_module(module_name)
    attach = getattr(module, attach_name)
    blend_config = getattr(module, blend_config_name)(
        min_pair_weight=0.0,
        max_pair_weight=0.0,
    )
    pair_config = CandidatePairForwardBackwardConfig(
        score_column="ranker_score",
        fallback_score_columns=(),
        score_normalization="rank",
        sigma_log_weight=0.0,
        output_score_column="test_pair_score",
    )
    rows = attach(
        _candidates(order),
        pair_config=pair_config,
        blend_config=blend_config,
    ).rows
    return rows.set_index("track_id").sort_index(), blend_config.output_score_column


@pytest.mark.parametrize(
    (
        "module_name",
        "attach_name",
        "blend_config_name",
        "local_score_column",
        "rank_column",
    ),
    _VARIANTS,
)
def test_adaptive_pair_rank_ties_are_permutation_invariant(
    module_name: str,
    attach_name: str,
    blend_config_name: str,
    local_score_column: str,
    rank_column: str,
) -> None:
    forward, output_score_column = _run_variant(
        module_name,
        attach_name,
        blend_config_name,
        ("tied-a", "tied-b", "lower"),
    )
    reversed_rows, reversed_output_column = _run_variant(
        module_name,
        attach_name,
        blend_config_name,
        ("lower", "tied-b", "tied-a"),
    )

    assert reversed_output_column == output_score_column
    for column in (local_score_column, output_score_column, rank_column):
        pd.testing.assert_series_equal(forward[column], reversed_rows[column])

    assert forward.loc["tied-a", local_score_column] == pytest.approx(
        forward.loc["tied-b", local_score_column]
    )
    assert forward.loc["tied-a", output_score_column] == pytest.approx(
        forward.loc["tied-b", output_score_column]
    )
    assert forward.loc["tied-a", rank_column] == pytest.approx(
        forward.loc["tied-b", rank_column]
    )
    assert forward.loc["tied-a", local_score_column] > forward.loc[
        "lower", local_score_column
    ]
