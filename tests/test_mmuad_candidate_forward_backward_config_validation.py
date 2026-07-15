from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_forward_backward import (
    CandidateForwardBackwardConfig,
    attach_forward_backward_candidate_prior,
)


NUMERIC_CONFIG_FIELDS = (
    "default_sigma_m",
    "sigma_min_m",
    "sigma_max_m",
    "score_weight",
    "sigma_log_weight",
    "transition_distance_std_m",
    "transition_speed_std_mps",
    "max_speed_mps",
    "speed_gate_penalty",
    "source_switch_penalty",
    "branch_switch_penalty",
    "track_continuation_bonus",
    "time_gap_penalty",
)


@pytest.mark.parametrize("field", NUMERIC_CONFIG_FIELDS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_forward_backward_rejects_nonfinite_controls_before_empty_return(
    field: str,
    value: float,
) -> None:
    config = CandidateForwardBackwardConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        attach_forward_backward_candidate_prior(pd.DataFrame(), config=config)


@pytest.mark.parametrize("value", [True, np.bool_(False), np.array([1.0])])
def test_forward_backward_rejects_boolean_and_nonscalar_controls(
    value: object,
) -> None:
    config = CandidateForwardBackwardConfig(score_weight=value)

    with pytest.raises(ValueError, match="score_weight"):
        attach_forward_backward_candidate_prior(pd.DataFrame(), config=config)


def test_forward_backward_accepts_finite_numpy_scalar_controls() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["track-1"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "confidence": [1.0],
        }
    )
    config = CandidateForwardBackwardConfig(
        default_sigma_m=np.array(10.0),
        sigma_min_m=np.float64(1.0),
        sigma_max_m=np.array(30.0),
        score_weight=np.float32(1.0),
        sigma_log_weight=np.int64(1),
        transition_distance_std_m=np.array(2.0),
        transition_speed_std_mps=np.float64(15.0),
        max_speed_mps=np.float32(80.0),
        speed_gate_penalty=np.array(25.0),
        source_switch_penalty=np.float64(0.25),
        branch_switch_penalty=np.float32(0.25),
        track_continuation_bonus=np.array(0.5),
        time_gap_penalty=np.int64(0),
    )

    augmented = attach_forward_backward_candidate_prior(candidates, config=config).rows

    assert augmented["candidate_forward_backward_score"].tolist() == pytest.approx([1.0])
