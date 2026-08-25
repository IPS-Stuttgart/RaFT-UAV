import numpy as np

from raft_uav.evaluation.metrics import (
    interpolate_positions_at_times,
    nearest_time_indices,
    position_errors_at_estimates_m,
    position_errors_m,
)


def test_nearest_time_indices_handles_large_finite_timestamp_gaps() -> None:
    reference_times = np.array([-1.0e308, 1.0e308])
    query_times = np.array([9.0e307])

    with np.errstate(over="raise", invalid="raise"):
        indices = nearest_time_indices(reference_times, query_times)

    np.testing.assert_array_equal(indices, np.array([1]))


def test_position_errors_skip_unused_overflowing_time_delta() -> None:
    truth_times = np.array([-1.0e308, -9.0e307])
    truth_positions = np.zeros((2, 3), dtype=float)
    estimate_times = np.array([1.0e308])
    estimate_positions = np.array([[3.0, 4.0, 0.0]])

    with np.errstate(over="raise", invalid="raise"):
        errors = position_errors_at_estimates_m(
            estimate_times,
            estimate_positions,
            truth_times,
            truth_positions,
            max_time_delta_s=None,
            dimensions=3,
        )
        gated = position_errors_at_estimates_m(
            estimate_times,
            estimate_positions,
            truth_times,
            truth_positions,
            max_time_delta_s=np.finfo(float).max,
            dimensions=3,
        )

    np.testing.assert_allclose(errors, np.array([5.0]))
    assert gated.size == 0


def test_time_tolerance_paths_handle_large_finite_endpoint_gaps() -> None:
    estimate_times = np.array([-1.0e308, 1.0e308])
    estimate_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    truth_times = np.array([1.0e308])
    truth_positions = np.array([[2.0, 0.0, 0.0]])

    with np.errstate(over="raise", invalid="raise"):
        interpolated, valid = interpolate_positions_at_times(
            estimate_times,
            estimate_positions,
            truth_times,
            max_time_delta_s=0.0,
        )
        errors = position_errors_m(
            estimate_times,
            estimate_positions,
            truth_times,
            truth_positions,
            max_time_delta_s=0.0,
            dimensions=3,
        )

    np.testing.assert_allclose(interpolated, truth_positions)
    np.testing.assert_array_equal(valid, np.array([True]))
    np.testing.assert_allclose(errors, np.array([0.0]))
