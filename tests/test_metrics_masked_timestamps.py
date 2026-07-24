import numpy as np

from raft_uav.evaluation.metrics import (
    interpolate_positions_at_times,
    position_errors_at_estimates_m,
    position_errors_at_times_m,
)


def test_interpolation_marks_masked_query_timestamps_invalid() -> None:
    reference_times = np.array([0.0, 1.0])
    reference_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    query_times = np.ma.array([0.25, 0.75], mask=[True, False])

    interpolated, valid = interpolate_positions_at_times(
        reference_times,
        reference_positions,
        query_times,
    )

    assert np.isnan(interpolated[0]).all()
    np.testing.assert_allclose(interpolated[1], np.array([0.75, 0.0, 0.0]))
    np.testing.assert_array_equal(valid, np.array([False, True]))


def test_position_errors_ignore_masked_estimate_timestamps() -> None:
    truth_times = np.array([0.0, 1.0])
    truth_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    estimate_times = np.ma.array([0.0, 1.0], mask=[True, False])
    estimate_positions = np.array(
        [
            [100.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    errors = position_errors_at_times_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
    )

    np.testing.assert_allclose(errors, np.array([0.0]))


def test_nearest_sample_metrics_ignore_masked_truth_timestamps() -> None:
    truth_times = np.ma.array([0.0, 1.0], mask=[True, False])
    truth_positions = np.array(
        [
            [100.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    estimate_times = np.array([0.0])
    estimate_positions = np.array([[1.0, 0.0, 0.0]])

    errors = position_errors_at_estimates_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
    )

    np.testing.assert_allclose(errors, np.array([0.0]))
