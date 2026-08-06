from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus import (
    TemporalConsensusConfig,
    add_temporal_candidate_consensus,
)
from raft_uav.mmuad.candidate_temporal_consensus_assignment import (
    add_assignment_temporal_candidate_consensus,
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
_ENTRYPOINTS: tuple[Callable[..., object], ...] = (
    add_temporal_candidate_consensus,
    add_assignment_temporal_candidate_consensus,
)


def _object_scalar(value: object) -> np.ndarray:
    boxed = np.empty((), dtype=object)
    boxed[()] = value
    return boxed


@pytest.mark.parametrize("config", [False, 0, "", {}, []])
def test_temporal_consensus_rejects_falsy_non_config_values(config: object) -> None:
    with pytest.raises(TypeError, match="TemporalConsensusConfig"):
        add_temporal_candidate_consensus(pd.DataFrame(), config=config)  # type: ignore[arg-type]


@pytest.mark.parametrize("config", [False, 0, "", {}, []])
def test_assignment_temporal_consensus_rejects_falsy_non_config_values(
    config: object,
) -> None:
    with pytest.raises(TypeError, match="TemporalConsensusConfig"):
        add_assignment_temporal_candidate_consensus(
            pd.DataFrame(),
            config=config,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field_name", _NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, 1.0 + 2.0j])
def test_temporal_consensus_rejects_invalid_numeric_config(
    field_name: str,
    value: object,
) -> None:
    config = replace(TemporalConsensusConfig(), **{field_name: value})

    with pytest.raises(ValueError, match=rf"{field_name} must be a finite real scalar"):
        add_temporal_candidate_consensus(pd.DataFrame(), config=config)


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
@pytest.mark.parametrize("field_name", _NUMERIC_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        _object_scalar(np.array(True)),
        _object_scalar(np.array([0.25])),
    ],
)
def test_temporal_consensus_rejects_nested_pseudo_scalars(
    entrypoint: Callable[..., object],
    field_name: str,
    value: object,
) -> None:
    config = replace(TemporalConsensusConfig(), **{field_name: value})

    with pytest.raises(ValueError, match=rf"{field_name} must be a finite real scalar"):
        entrypoint(pd.DataFrame(), config=config)


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_temporal_consensus_rejects_cyclic_object_scalars(
    entrypoint: Callable[..., object],
) -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    config = replace(TemporalConsensusConfig(), max_speed_mps=cyclic)

    with pytest.raises(ValueError, match="max_speed_mps must be a finite real scalar"):
        entrypoint(pd.DataFrame(), config=config)


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_temporal_consensus_accepts_recursively_boxed_real_scalars(
    entrypoint: Callable[..., object],
) -> None:
    config = replace(
        TemporalConsensusConfig(),
        max_speed_mps=_object_scalar(_object_scalar(np.float64(60.0))),
        base_score_weight=_object_scalar(np.array(0.25)),
    )

    result = entrypoint(pd.DataFrame(), config=config)

    assert result.rows.empty


def test_temporal_consensus_accepts_valid_explicit_config_on_empty_input() -> None:
    result = add_temporal_candidate_consensus(
        pd.DataFrame(),
        config=TemporalConsensusConfig(),
    )

    assert result.rows.empty
