from __future__ import annotations

from dataclasses import replace
import math

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    attach_pair_forward_backward_candidate_prior,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_sigma_m", math.nan),
        ("sigma_min_m", math.inf),
        ("sigma_max_m", -math.inf),
        ("score_weight", math.nan),
        ("sigma_log_weight", math.inf),
        ("transition_distance_std_m", math.nan),
        ("transition_speed_std_mps", math.inf),
        ("max_speed_mps", math.nan),
        ("speed_gate_penalty", math.inf),
        ("acceleration_std_mps2", math.nan),
        ("max_acceleration_mps2", math.inf),
        ("acceleration_gate_penalty", math.nan),
        ("source_switch_penalty", math.inf),
        ("branch_switch_penalty", math.nan),
        ("track_continuation_bonus", math.inf),
        ("time_gap_penalty", math.nan),
    ],
)
def test_pair_forward_backward_rejects_nonfinite_controls(
    field: str,
    value: float,
) -> None:
    config = replace(
        CandidatePairForwardBackwardConfig(),
        **{field: value},
    )

    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        attach_pair_forward_backward_candidate_prior(
            pd.DataFrame(),
            config=config,
        )
