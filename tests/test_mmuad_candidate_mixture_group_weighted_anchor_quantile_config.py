from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_weighted_anchor_quantile import (
    add_weighted_quantile_multi_anchor_conditioned_selection_utility,
    run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map,
    select_weighted_quantile_posterior_mass_hypothesis_group_topk,
)


@pytest.mark.parametrize(
    "field",
    ["mixture_config", "anchor_config", "quantile_config"],
)
def test_weighted_quantile_add_rejects_falsy_invalid_configs(field: str) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be .* or None"):
        add_weighted_quantile_multi_anchor_conditioned_selection_utility(
            pd.DataFrame(),
            {},
            **{field: False},
        )


@pytest.mark.parametrize(
    "field",
    [
        "mixture_config",
        "group_config",
        "selection_config",
        "anchor_config",
        "quantile_config",
    ],
)
def test_weighted_quantile_select_rejects_falsy_invalid_configs(field: str) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be .* or None"):
        select_weighted_quantile_posterior_mass_hypothesis_group_topk(
            pd.DataFrame(),
            anchor_estimates={},
            **{field: False},
        )


@pytest.mark.parametrize(
    "field",
    [
        "mixture_config",
        "group_config",
        "selection_config",
        "anchor_config",
        "quantile_config",
    ],
)
def test_weighted_quantile_run_rejects_falsy_invalid_configs(field: str) -> None:
    with pytest.raises(TypeError, match=rf"{field} must be .* or None"):
        run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map(
            pd.DataFrame(),
            anchor_estimates={},
            **{field: False},
        )
