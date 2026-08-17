from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.imm import (
    AsyncInteractingMultipleModelTracker,
    uniform_ctmc_transition_matrix,
)


@pytest.mark.parametrize("dt_s", [np.nan, np.inf, -np.inf])
def test_uniform_ctmc_transition_rejects_nonfinite_dt(dt_s: float) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        uniform_ctmc_transition_matrix(
            3,
            dt_s=dt_s,
            mode_switch_time_constant_s=20.0,
        )


@pytest.mark.parametrize(
    "time_constant_s",
    [np.nan, np.inf, -np.inf, 0.0, -1.0],
)
def test_uniform_ctmc_transition_rejects_invalid_time_constant(
    time_constant_s: float,
) -> None:
    with pytest.raises(ValueError, match="mode_switch_time_constant_s"):
        uniform_ctmc_transition_matrix(
            3,
            dt_s=1.0,
            mode_switch_time_constant_s=time_constant_s,
        )


def test_uniform_ctmc_transition_preserves_negative_dt_clamp() -> None:
    transition = uniform_ctmc_transition_matrix(
        3,
        dt_s=-1.0,
        mode_switch_time_constant_s=20.0,
    )

    np.testing.assert_allclose(transition, np.eye(3), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("time_constant_s", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_imm_tracker_rejects_invalid_time_constant_at_construction(
    time_constant_s: float,
) -> None:
    with pytest.raises(ValueError, match="mode_switch_time_constant_s"):
        AsyncInteractingMultipleModelTracker(
            np.zeros(3),
            0.0,
            mode_switch_time_constant_s=time_constant_s,
        )
