import numpy as np

from raft_uav.evaluation.metrics import position_errors_m


def test_single_sample_metric_rejects_truth_outside_zero_width_support() -> None:
    truth_times = np.array([0.0, 10.0])
    truth_positions = np.column_stack(
        [truth_times, np.zeros_like(truth_times), np.zeros_like(truth_times)]
    )

    errors = position_errors_m(
        np.array([5.0]),
        np.array([[5.0, 0.0, 0.0]]),
        truth_times,
        truth_positions,
        max_time_delta_s=10.0,
    )

    assert errors.size == 0


def test_single_sample_metric_scores_exact_truth_grid_timestamp() -> None:
    truth_times = np.array([0.0, 5.0, 10.0])
    truth_positions = np.column_stack(
        [truth_times, np.zeros_like(truth_times), np.zeros_like(truth_times)]
    )

    errors = position_errors_m(
        np.array([5.0]),
        np.array([[7.0, 0.0, 0.0]]),
        truth_times,
        truth_positions,
        max_time_delta_s=0.0,
    )

    np.testing.assert_allclose(errors, np.array([2.0]))
