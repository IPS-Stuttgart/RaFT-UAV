import numpy as np

from raft_uav.evaluation.metrics import (
    interpolate_positions_at_times,
    position_errors_m,
    summarize_errors,
)


def test_position_errors_ignore_masked_trajectory_samples():
    times = np.array([0.0, 1.0, 2.0])
    truth_positions = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times)]
    )
    estimate_positions = np.ma.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        mask=[
            [False, False, False],
            [True, True, True],
            [False, False, False],
        ],
    )

    errors = position_errors_m(
        times,
        estimate_positions,
        times,
        truth_positions,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.zeros(3))


def test_interpolation_preserves_reference_and_query_masks():
    reference_times = np.array([0.0, 1.0, 2.0])
    reference_positions = np.ma.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        mask=[
            [False, False, False],
            [True, True, True],
            [False, False, False],
        ],
    )
    query_times = np.ma.array([1.0, 1.5], mask=[False, True])

    interpolated, valid = interpolate_positions_at_times(
        reference_times,
        reference_positions,
        query_times,
    )

    np.testing.assert_allclose(interpolated[0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(valid, np.array([True, False]))
    assert np.isnan(interpolated[1]).all()


def test_summarize_errors_ignores_masked_values():
    summary = summarize_errors(
        np.ma.array([1.0, 100.0, 3.0], mask=[False, True, False])
    )

    assert summary["count"] == 2.0
    assert summary["mean_m"] == 2.0
    assert summary["rmse_m"] == np.sqrt(5.0)
    assert summary["max_m"] == 3.0
