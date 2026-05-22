import json

import numpy as np

from raft_uav.evaluation.metrics import (
    position_errors_at_estimates_m,
    position_errors_m,
    sampled_position_errors_m,
    summarize_errors,
)


def test_metrics_return_finite_summaries_on_synthetic_trajectories():
    truth_times = np.array([0.0, 1.0, 2.0])
    truth_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    estimate_times = np.array([0.0, 1.0, 2.0])
    estimate_positions = truth_positions + np.array([0.0, 0.5, 0.25])

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=0.1,
        dimensions=3,
    )
    summary = summarize_errors(errors)

    assert summary["count"] == 3.0
    assert np.isfinite(summary["mean_m"])
    assert np.isfinite(summary["std_m"])
    assert np.isfinite(summary["rmse_m"])
    assert np.isfinite(summary["mae_m"])
    assert np.isfinite(summary["p95_m"])
    assert np.isfinite(summary["max_m"])


def test_position_errors_interpolate_estimates_to_truth_time_grid():
    truth_times = np.array([0.0, 1.0, 2.0, 3.0])
    truth_positions = np.column_stack(
        [truth_times, np.zeros_like(truth_times), np.zeros_like(truth_times)]
    )
    estimate_times = np.array([0.0, 3.0])
    estimate_positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=2.0,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.ones(4))


def test_sampled_position_errors_use_nearest_truth_samples_and_time_gate():
    truth_times = np.array([0.0, 10.0, 20.0])
    truth_positions = np.column_stack(
        [truth_times, np.zeros_like(truth_times), np.zeros_like(truth_times)]
    )
    estimate_times = np.array([0.2, 9.8, 30.0])
    estimate_positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    )

    errors = sampled_position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=0.5,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.array([1.0, 2.0]))


def test_position_errors_skip_truth_samples_without_local_estimate_support():
    truth_times = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    truth_positions = np.column_stack(
        [truth_times, np.zeros_like(truth_times), np.zeros_like(truth_times)]
    )
    estimate_times = np.array([0.0, 10.0])
    estimate_positions = truth_positions[[0, -1], :]

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=2.0,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.zeros(2))


def test_position_errors_are_order_and_duplicate_timestamp_stable():
    truth_times = np.array([0.0, 1.0])
    truth_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    estimate_times = np.array([1.0, 0.0, 0.0])
    estimate_positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [99.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.zeros(2))


def test_position_errors_at_estimates_uses_nearest_truth_and_time_gate():
    truth_times = np.array([0.0, 10.0])
    truth_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    estimate_times = np.array([0.0, 5.0, 10.0])
    estimate_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )

    errors = position_errors_at_estimates_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=10.0,
        dimensions=3,
    )
    gated = position_errors_at_estimates_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=2.0,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.array([0.0, 5.0, 0.0]))
    np.testing.assert_allclose(gated, np.array([0.0, 0.0]))


def test_position_errors_at_estimates_preserves_duplicate_estimate_times():
    truth_times = np.array([0.0])
    truth_positions = np.array([[0.0, 0.0, 0.0]])
    estimate_times = np.array([0.0, 0.0])
    estimate_positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    errors = position_errors_at_estimates_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.array([1.0, 2.0]))


def test_summarize_errors_empty_input_is_strict_json_compatible():
    summary = summarize_errors(np.array([]))

    assert summary["count"] == 0.0
    assert summary["mean_m"] is None
    assert summary["std_m"] is None
    assert summary["rmse_m"] is None
    assert summary["mae_m"] is None
    assert summary["p50_m"] is None
    assert summary["p95_m"] is None
    assert summary["max_m"] is None
    json.dumps(summary, allow_nan=False)


def test_summarize_errors_ignores_non_finite_values():
    summary = summarize_errors(np.array([3.0, np.nan, np.inf, 4.0]))

    assert summary["count"] == 2.0
    assert summary["mean_m"] == 3.5
    assert summary["mae_m"] == 3.5
    assert summary["std_m"] == 0.5
    assert summary["max_m"] == 4.0
    json.dumps(summary, allow_nan=False)
