from __future__ import annotations

import math
import runpy
import sys

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


@pytest.mark.parametrize("field_name", _NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_temporal_consensus_rejects_nonfinite_config(
    field_name: str,
    value: float,
) -> None:
    config = TemporalConsensusConfig(**{field_name: value})

    with pytest.raises(ValueError, match=rf"{field_name} must be finite"):
        add_temporal_candidate_consensus(pd.DataFrame(), config=config)


def test_temporal_consensus_package_supports_python_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "raft_uav.mmuad.candidate_temporal_consensus"
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")

    assert exc_info.value.code == 0
