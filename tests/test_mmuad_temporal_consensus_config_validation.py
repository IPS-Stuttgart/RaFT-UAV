from __future__ import annotations

from dataclasses import replace
import math

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus import (
    TemporalConsensusConfig,
    add_temporal_candidate_consensus,
)

_NUMERIC_FIELDS = (
    "max_time_gap_s",
    "max_speed_mps",
    "distance_scale_m",
    "acceleration_scale_mps2",
    "base_score_weight",
    "backward_support_weight",
    "forward_support_weight",
    "bidirectional_bonus",
    "interpolation_weight",
    "acceleration_weight",
    "source_diversity_bonus",
    "branch_diversity_bonus",
)


@pytest.mark.parametrize("config", [False, 0, "", {}, []])
def test_temporal_consensus_rejects_falsy_non_config_values(config: object) -> None:
    with pytest.raises(TypeError, match="TemporalConsensusConfig"):
        add_temporal_candidate_consensus(pd.DataFrame(), config=config)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", _NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, 1.0 + 2.0j])
def test_temporal_consensus_rejects_invalid_numeric_config(
    field_name: str,
    value: object,
) -> None:
    config = replace(TemporalConsensusConfig(), **{field_name: value})

    with pytest.raises(ValueError, match=rf"{field_name} must be a finite real scalar"):
        add_temporal_candidate_consensus(pd.DataFrame(), config=config)


def test_temporal_consensus_accepts_valid_explicit_config_on_empty_input() -> None:
    result = add_temporal_candidate_consensus(
        pd.DataFrame(),
        config=TemporalConsensusConfig(),
    )

    assert result.rows.empty
