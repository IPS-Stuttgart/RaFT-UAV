from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
import pytest

from raft_uav.evaluation.metrics import (
    empirical_position_covariance_at_times,
    interpolate_positions_at_times,
    position_errors_at_estimates_m,
    position_errors_at_times_m,
    position_errors_m,
    sampled_position_errors_m,
)

_TIMES = np.asarray([0.0, 1.0], dtype=float)
_POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)


def _call_position_errors_m(max_time_delta_s: float) -> object:
    return position_errors_m(
        _TIMES,
        _POSITIONS,
        _TIMES,
        _POSITIONS,
        max_time_delta_s=max_time_delta_s,
    )


def _call_position_errors_at_estimates_m(max_time_delta_s: float) -> object:
    return position_errors_at_estimates_m(
        _TIMES,
        _POSITIONS,
        _TIMES,
        _POSITIONS,
        max_time_delta_s=max_time_delta_s,
    )


def _call_sampled_position_errors_m(max_time_delta_s: float) -> object:
    return sampled_position_errors_m(
        _TIMES,
        _POSITIONS,
        _TIMES,
        _POSITIONS,
        max_time_delta_s=max_time_delta_s,
    )


def _call_interpolate_positions_at_times(max_time_delta_s: float) -> object:
    return interpolate_positions_at_times(
        _TIMES,
        _POSITIONS,
        _TIMES,
        max_time_delta_s=max_time_delta_s,
    )


def _call_position_errors_at_times_m(max_time_delta_s: float) -> object:
    return position_errors_at_times_m(
        _TIMES,
        _POSITIONS,
        _TIMES,
        _POSITIONS,
        max_time_delta_s=max_time_delta_s,
    )


def _call_empirical_position_covariance_at_times(max_time_delta_s: float) -> object:
    return empirical_position_covariance_at_times(
        _TIMES,
        _POSITIONS,
        _TIMES,
        _POSITIONS,
        max_time_delta_s=max_time_delta_s,
    )


_METRIC_CALLS: tuple[Callable[[float], object], ...] = (
    _call_position_errors_m,
    _call_position_errors_at_estimates_m,
    _call_sampled_position_errors_m,
    _call_interpolate_positions_at_times,
    _call_position_errors_at_times_m,
    _call_empirical_position_covariance_at_times,
)


@pytest.mark.parametrize("metric_call", _METRIC_CALLS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.1])
def test_public_metric_apis_reject_invalid_time_gates(
    metric_call: Callable[[float], object],
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_time_delta_s must be finite and non-negative",
    ):
        metric_call(value)


def test_zero_time_gate_remains_valid_for_exact_timestamp_matches() -> None:
    errors = position_errors_m(
        _TIMES,
        _POSITIONS,
        _TIMES,
        _POSITIONS,
        max_time_delta_s=0.0,
    )

    np.testing.assert_allclose(errors, np.zeros(2))
