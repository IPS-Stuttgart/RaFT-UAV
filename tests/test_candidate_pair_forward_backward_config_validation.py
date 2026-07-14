from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    attach_pair_forward_backward_candidate_prior,
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
        "acceleration_std_mps2",
        "max_acceleration_mps2",
        "acceleration_gate_penalty",
        "source_switch_penalty",
        "branch_switch_penalty",
        "track_continuation_bonus",
        "time_gap_penalty",
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_pair_forward_backward_rejects_nonfinite_numeric_controls(
    field: str,
    value: float,
) -> None:
    config = CandidatePairForwardBackwardConfig(**{field: value})

    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        attach_pair_forward_backward_candidate_prior(pd.DataFrame(), config=config)
