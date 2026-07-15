from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.imm import fixed_turn_rate_matrix, uniform_ctmc_transition_matrix


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_uniform_ctmc_transition_matrix_rejects_nonfinite_time_steps(value: float) -> None:
    with pytest.raises(ValueError, match="dt_s must be a finite scalar"):
        uniform_ctmc_transition_matrix(
            n_modes=3,
            dt_s=value,
            mode_switch_time_constant_s=20.0,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_uniform_ctmc_transition_matrix_rejects_nonfinite_time_constants(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="mode_switch_time_constant_s must be a finite scalar"):
        uniform_ctmc_transition_matrix(
            n_modes=3,
            dt_s=1.0,
            mode_switch_time_constant_s=value,
        )


@pytest.mark.parametrize("dt_s", [np.nan, np.inf, -np.inf])
def test_fixed_turn_rate_matrix_rejects_nonfinite_time_steps(dt_s: float) -> None:
    with pytest.raises(ValueError, match="dt_s must be a finite scalar"):
        fixed_turn_rate_matrix(dt_s, turn_rate_radps=0.1)


@pytest.mark.parametrize("turn_rate", [np.nan, np.inf, -np.inf])
def test_fixed_turn_rate_matrix_rejects_nonfinite_turn_rates(turn_rate: float) -> None:
    with pytest.raises(ValueError, match="turn_rate_radps must be a finite scalar"):
        fixed_turn_rate_matrix(1.0, turn_rate_radps=turn_rate)


def test_uniform_ctmc_transition_matrix_preserves_finite_negative_dt_behavior() -> None:
    matrix = uniform_ctmc_transition_matrix(
        n_modes=3,
        dt_s=-1.0,
        mode_switch_time_constant_s=20.0,
    )

    np.testing.assert_allclose(matrix, np.eye(3))
