from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.imm import IMMMode


def test_imm_cv_mode_uses_validated_transition_helper() -> None:
    mode = IMMMode("cv", acceleration_std_mps2=4.0, turn_rate_radps=0.0)

    with pytest.raises(
        ValueError,
        match="dt_s must be a finite, non-negative real scalar",
    ):
        mode.transition_matrix(np.inf)


def test_imm_cv_mode_uses_validated_process_noise_helper() -> None:
    mode = IMMMode("cv", acceleration_std_mps2=4.0, turn_rate_radps=0.0)

    with pytest.raises(
        ValueError,
        match="dt_s must be a finite, non-negative real scalar",
    ):
        mode.process_noise(-1.0)


@pytest.mark.parametrize(
    "turn_rate_radps",
    [np.deg2rad(6.0), -np.deg2rad(6.0)],
)
def test_imm_turn_modes_reject_negative_transition_interval(
    turn_rate_radps: float,
) -> None:
    mode = IMMMode(
        "turn",
        acceleration_std_mps2=4.0,
        turn_rate_radps=turn_rate_radps,
    )

    with pytest.raises(
        ValueError,
        match="dt_s must be a finite, non-negative real scalar",
    ):
        mode.transition_matrix(-1.0)
