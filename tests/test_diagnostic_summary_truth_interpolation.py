from __future__ import annotations

import pandas as pd

from raft_uav.evaluation.diagnostics import build_diagnostic_summary


def test_worst_time_windows_interpolate_truth_at_estimate_timestamps():
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    estimates = pd.DataFrame(
        {
            "time_s": [5.0],
            "source": ["radar"],
            "track_id": [7],
            "residual_norm_m": [0.0],
            "covariance_scale": [1.0],
            "east_m": [5.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    summary = build_diagnostic_summary(
        estimate_frame=estimates,
        selected_radar=pd.DataFrame(),
        truth=truth,
        max_eval_time_delta_s=5.0,
        window_s=10.0,
    )

    assert summary["worst_time_windows"] == [
        {
            "time_start_s": 0.0,
            "time_end_s": 10.0,
            "count": 1,
            "rmse_3d_m": 0.0,
            "mae_3d_m": 0.0,
            "p95_3d_m": 0.0,
            "max_3d_m": 0.0,
            "mean_residual_norm_m": 0.0,
            "covariance_inflation_count": 0,
            "track_switch_count": 0,
        }
    ]
