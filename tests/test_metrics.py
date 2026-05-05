import numpy as np

from raft_uav.evaluation.metrics import position_errors_m, summarize_errors


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
    assert np.isfinite(summary["rmse_m"])
    assert np.isfinite(summary["mae_m"])
    assert np.isfinite(summary["p95_m"])
