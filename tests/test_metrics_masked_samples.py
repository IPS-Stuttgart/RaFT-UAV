import numpy as np

from raft_uav.evaluation.metrics import (
    interpolate_positions_at_times,
    position_errors_at_estimates_m,
)


def _trajectory() -> tuple[np.ndarray, np.ndarray]:
    times = np.array([0.0, 1.0])
    positions = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times)]
    )
    return times, positions


def test_interpolation_marks_masked_query_timestamps_invalid() -> None:
    times, positions = _trajectory()
    query = np.ma.array([0.5, 0.75], mask=[False, True])

    interpolated, valid = interpolate_positions_at_times(
        times,
        positions,
        query,
    )

    np.testing.assert_allclose(interpolated[0], np.array([0.5, 0.0, 0.0]))
    assert np.isnan(interpolated[1]).all()
    np.testing.assert_array_equal(valid, np.array([True, False]))


def test_position_errors_ignore_masked_timestamp_rows() -> None:
    truth_times, truth_positions = _trajectory()
    estimate_times = np.ma.array([0.0, 1.0], mask=[False, True])
    estimate_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )

    errors = position_errors_at_estimates_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
    )

    np.testing.assert_allclose(errors, np.array([0.0]))


def test_position_errors_ignore_masked_position_rows() -> None:
    truth_times, truth_positions = _trajectory()
    estimate_positions = np.ma.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        mask=[
            [False, False, False],
            [True, True, True],
        ],
    )

    errors = position_errors_at_estimates_m(
        truth_times,
        estimate_positions,
        truth_times,
        truth_positions,
    )

    np.testing.assert_allclose(errors, np.array([0.0]))
