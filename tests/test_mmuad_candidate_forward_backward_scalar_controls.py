from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_forward_backward import (
    CandidateForwardBackwardConfig,
    attach_forward_backward_candidate_prior,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 1.0],
            "source": ["radar", "radar"],
            "track_id": ["track", "track"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [1.0, 1.0],
        }
    )


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.array(True),
        1.0 + 0.0j,
        np.complex128(1.0 + 0.0j),
        np.array([1.0]),
    ],
)
def test_forward_backward_rejects_lossy_numeric_config_controls(
    field: str,
    value: object,
) -> None:
    config = replace(CandidateForwardBackwardConfig(), **{field: value})

    with pytest.raises(
        ValueError,
        match=rf"^{field} must be a non-Boolean real scalar$",
    ):
        attach_forward_backward_candidate_prior(_candidates(), config=config)


def test_forward_backward_accepts_zero_dimensional_real_controls() -> None:
    config = replace(
        CandidateForwardBackwardConfig(),
        score_weight=np.array(1.0),
        transition_distance_std_m=np.array(2.0),
    )

    result = attach_forward_backward_candidate_prior(_candidates(), config=config).rows

    assert result["candidate_forward_backward_score"].notna().all()
