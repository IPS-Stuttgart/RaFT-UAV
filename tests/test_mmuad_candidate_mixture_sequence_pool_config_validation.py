from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map_sequence_pool_selector import (
    CandidatePoolSequenceSelectorConfig,
    build_sequence_candidate_pool_variants,
    run_sequence_pool_selector,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["track-1"],
            "candidate_branch": ["raw"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )


@pytest.mark.parametrize("config", [False, 0, "", {}, []])
def test_pool_builder_rejects_falsey_non_config_values(config: object) -> None:
    with pytest.raises(TypeError, match="CandidatePoolSequenceSelectorConfig"):
        build_sequence_candidate_pool_variants(_candidates(), config=config)


@pytest.mark.parametrize("config", [False, 0, "", {}, []])
def test_pool_selector_rejects_falsey_non_config_values(config: object) -> None:
    with pytest.raises(TypeError, match="CandidatePoolSequenceSelectorConfig"):
        run_sequence_pool_selector(_candidates(), selector_config=config)


@pytest.mark.parametrize("config", [False, 0, "", {}, []])
def test_pool_selector_rejects_falsey_non_mixture_configs(config: object) -> None:
    with pytest.raises(TypeError, match="CandidateMixtureMapConfig"):
        run_sequence_pool_selector(_candidates(), mixture_config=config)


@pytest.mark.parametrize(
    "value",
    [True, 1.5, "1.5", np.nan, np.inf, 1.0 + 0.0j, np.array([1])],
)
def test_selector_rejects_lossy_max_leave_one_out(value: object) -> None:
    config = replace(
        CandidatePoolSequenceSelectorConfig(),
        max_leave_one_out=value,
    )

    with pytest.raises(ValueError, match="max_leave_one_out"):
        build_sequence_candidate_pool_variants(_candidates(), config=config)


@pytest.mark.parametrize(
    "value",
    [True, -0.1, 1.1, np.nan, np.inf, 0.5 + 0.0j, np.array([0.5])],
)
def test_selector_rejects_invalid_group_frame_fraction(value: object) -> None:
    config = replace(
        CandidatePoolSequenceSelectorConfig(),
        min_group_frame_fraction=value,
    )

    with pytest.raises(ValueError, match="min_group_frame_fraction"):
        build_sequence_candidate_pool_variants(_candidates(), config=config)


@pytest.mark.parametrize(
    "field_name",
    [
        "include_full_pool",
        "include_leave_one_out",
        "restore_missing_frames",
        "normalize_component_count",
    ],
)
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_selector_rejects_non_boolean_flags(
    field_name: str,
    value: object,
) -> None:
    config = replace(
        CandidatePoolSequenceSelectorConfig(),
        **{field_name: value},
    )

    with pytest.raises(ValueError, match=rf"{field_name} must be a Boolean"):
        build_sequence_candidate_pool_variants(_candidates(), config=config)


def test_selector_accepts_exact_numpy_scalar_controls() -> None:
    config = CandidatePoolSequenceSelectorConfig(
        max_leave_one_out=np.int64(1),
        min_group_frame_fraction=np.float64(0.0),
        include_full_pool=np.bool_(True),
        include_leave_one_out=np.bool_(False),
        restore_missing_frames=np.bool_(True),
        normalize_component_count=np.bool_(True),
    )

    pools = build_sequence_candidate_pool_variants(_candidates(), config=config)

    assert set(pools) == {"full_pool"}
