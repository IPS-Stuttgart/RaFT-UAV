from __future__ import annotations

import numpy as np
import pytest

from raft_uav.evaluation.metrics import (
    nearest_time_indices,
    position_errors_at_estimates_m,
    summarize_errors,
)


def _boxed(value: object, *, depth: int = 2) -> np.ndarray:
    result = value
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = result
        result = wrapper
    return result


def test_nearest_time_indices_rejects_recursively_boxed_boolean_query() -> None:
    query_times = np.empty(1, dtype=object)
    query_times[0] = _boxed(True)

    with pytest.raises(
        ValueError,
        match="query_times_s must not contain Boolean values",
    ):
        nearest_time_indices(np.array([0.0, 1.0]), query_times)


def test_position_errors_rejects_recursively_boxed_complex_coordinate() -> None:
    times = np.array([0.0, 1.0])
    truth_positions = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times)]
    )
    estimate_positions = truth_positions.astype(object)
    estimate_positions[1, 0] = _boxed(np.complex64(1.0 + 4.0j))

    with pytest.raises(ValueError, match="positions_m must contain only real values"):
        position_errors_at_estimates_m(
            times,
            estimate_positions,
            times,
            truth_positions,
        )


def test_summarize_errors_rejects_recursively_boxed_boolean() -> None:
    errors = np.empty(1, dtype=object)
    errors[0] = _boxed(np.bool_(True))

    with pytest.raises(ValueError, match="errors_m must not contain Boolean values"):
        summarize_errors(errors)


def test_nearest_time_indices_accepts_recursively_boxed_real_query() -> None:
    query_times = np.empty(1, dtype=object)
    query_times[0] = _boxed(np.float64(1.0))

    indices = nearest_time_indices(np.array([0.0, 1.0]), query_times)

    np.testing.assert_array_equal(indices, np.array([1]))
