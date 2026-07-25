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


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.array([1.0]),
        1.0 + 0.0j,
        np.ma.masked,
        np.ma.array(1.0, mask=True),
    ],
)
@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("dt_s", "dt_s must be a finite scalar"),
        (
            "mode_switch_time_constant_s",
            "mode_switch_time_constant_s must be a finite scalar",
        ),
    ],
)
def test_uniform_ctmc_transition_matrix_rejects_malformed_scalar_controls(
    value: object,
    field: str,
    expected_message: str,
) -> None:
    kwargs: dict[str, object] = {
        "n_modes": 3,
        "dt_s": 1.0,
        "mode_switch_time_constant_s": 20.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=expected_message):
        uniform_ctmc_transition_matrix(**kwargs)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.array([1.0]),
        1.0 + 0.0j,
        np.ma.masked,
        np.ma.array(1.0, mask=True),
    ],
)
@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("dt_s", "dt_s must be a finite scalar"),
        ("turn_rate_radps", "turn_rate_radps must be a finite scalar"),
    ],
)
def test_fixed_turn_rate_matrix_rejects_malformed_scalar_controls(
    value: object,
    field: str,
    expected_message: str,
) -> None:
    kwargs: dict[str, object] = {"dt_s": 1.0, "turn_rate_radps": 0.1}
    kwargs[field] = value

    with pytest.raises(ValueError, match=expected_message):
        fixed_turn_rate_matrix(**kwargs)


@pytest.mark.parametrize(
    "n_modes",
    [
        True,
        np.bool_(True),
        0,
        -1,
        2.5,
        np.nan,
        np.inf,
        3.0 + 0.0j,
        np.array([3]),
        np.ma.masked,
        np.ma.array(3, mask=True),
    ],
)
def test_uniform_ctmc_transition_matrix_rejects_invalid_mode_counts(
    n_modes: object,
) -> None:
    with pytest.raises(ValueError, match="n_modes must be a positive integer"):
        uniform_ctmc_transition_matrix(
            n_modes=n_modes,
            dt_s=1.0,
            mode_switch_time_constant_s=20.0,
        )


@pytest.mark.parametrize("n_modes", [3, np.int64(3), 3.0, np.array(3), "3"])
def test_uniform_ctmc_transition_matrix_accepts_exact_scalar_mode_counts(
    n_modes: object,
) -> None:
    matrix = uniform_ctmc_transition_matrix(
        n_modes=n_modes,
        dt_s=1.0,
        mode_switch_time_constant_s=20.0,
    )

    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3))


def test_uniform_ctmc_transition_matrix_preserves_finite_negative_dt_behavior() -> None:
    matrix = uniform_ctmc_transition_matrix(
        n_modes=3,
        dt_s=-1.0,
        mode_switch_time_constant_s=20.0,
    )

    np.testing.assert_allclose(matrix, np.eye(3))
