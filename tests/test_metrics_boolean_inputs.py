from __future__ import annotations

import numpy as np
import pytest

from raft_uav.evaluation.metrics import (
    interpolate_positions_at_times,
    nearest_time_indices,
    position_errors_m,
    summarize_errors,
)


def _trajectory() -> tuple[np.ndarray, np.ndarray]:
    times = np.array([0.0, 1.0])
    positions = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times)]
    )
    return times, positions


def test_nearest_time_indices_rejects_boolean_reference_times() -> None:
    with pytest.raises(
        ValueError,
        match="reference_times_s must not contain Boolean values",
    ):
        nearest_time_indices(np.array([False, True]), np.array([0.25]))


def test_nearest_time_indices_rejects_boolean_hidden_by_numeric_promotion() -> None:
    with pytest.raises(
        ValueError,
        match="reference_times_s must not contain Boolean values",
    ):
        nearest_time_indices([False, 2], np.array([0.25]))


def test_nearest_time_indices_rejects_object_wrapped_boolean_queries() -> None:
    with pytest.raises(
        ValueError,
        match="query_times_s must not contain Boolean values",
    ):
        nearest_time_indices(
            np.array([0.0, 1.0]),
            np.array([True], dtype=object),
        )


def test_position_errors_rejects_boolean_trajectory_times() -> None:
    times, positions = _trajectory()

    with pytest.raises(ValueError, match="times_s must not contain Boolean values"):
        position_errors_m(
            np.array([False, True]),
            positions,
            times,
            positions,
        )


def test_position_errors_rejects_object_wrapped_boolean_positions() -> None:
    times, positions = _trajectory()
    invalid_positions = positions.astype(object)
    invalid_positions[0, 0] = False

    with pytest.raises(
        ValueError,
        match="positions_m must not contain Boolean values",
    ):
        position_errors_m(
            times,
            invalid_positions,
            times,
            positions,
        )


def test_interpolation_rejects_boolean_query_times() -> None:
    times, positions = _trajectory()

    with pytest.raises(
        ValueError,
        match="query_times_s must not contain Boolean values",
    ):
        interpolate_positions_at_times(
            times,
            positions,
            np.array([False, True]),
        )


def test_summarize_errors_rejects_boolean_values() -> None:
    with pytest.raises(ValueError, match="errors_m must not contain Boolean values"):
        summarize_errors(np.array([False, True]))


def test_metrics_continue_to_accept_integer_numeric_arrays() -> None:
    indices = nearest_time_indices(np.array([0, 1]), np.array([1]))
    summary = summarize_errors(np.array([0, 1]))

    np.testing.assert_array_equal(indices, np.array([1]))
    assert summary["count"] == 2.0
    assert summary["mean_m"] == 0.5
