from __future__ import annotations

import math

import pandas as pd

from raft_uav.evaluation.diagnostics import build_diagnostic_summary


def test_diagnostic_summary_reports_residuals_switches_inflation_and_windows():
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 10.0, 20.0, 30.0],
            "east_m": [0.0, 10.0, 20.0, 30.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
        }
    )
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, 10.0, 20.0, 30.0],
            "source": ["rf", "radar", "radar", "radar"],
            "track_id": [None, 10, 10, 11],
            "measurement_dim": [2, 3, 3, 3],
            "accepted": [True, True, True, True],
            "update_action": ["updated", "inflated", "updated", "inflated"],
            "residual_norm_m": [5.0, 50.0, 10.0, 100.0],
            "nis": [0.5, 20.0, 2.0, 40.0],
            "gate_threshold": [9.0, 11.0, 11.0, 11.0],
            "covariance_scale": [1.0, 2.0, 1.0, 4.0],
            "inflation_alpha": [0.5, 0.5, 0.5, 0.5],
            "east_m": [0.0, 10.0, 120.0, 130.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
        }
    )
    selected_radar = pd.DataFrame(
        {
            "time_s": [10.0, 20.0, 30.0],
            "track_id": [10, 10, 11],
        }
    )

    summary = build_diagnostic_summary(
        estimate_frame=estimates,
        selected_radar=selected_radar,
        truth=truth,
        max_eval_time_delta_s=1.0,
        top_n=3,
        window_s=20.0,
    )

    assert summary["top_residuals"][0]["residual_norm_m"] == 100.0
    assert summary["top_residuals"][0]["track_id"] == 11
    assert summary["track_switches"]["posterior_radar"]["count"] == 1
    assert summary["track_switches"]["selected_radar"]["count"] == 1
    assert summary["covariance_inflation"]["count"] == 2
    assert summary["covariance_inflation"]["by_source"] == {"radar": 2}
    assert summary["covariance_inflation"]["max_scale"] == 4.0
    assert summary["worst_time_windows"][0]["time_start_s"] == 20.0
    assert summary["worst_time_windows"][0]["count"] == 2
    assert math.isclose(summary["worst_time_windows"][0]["rmse_3d_m"], 100.0)


def test_diagnostic_summary_handles_empty_optional_columns():
    summary = build_diagnostic_summary(
        estimate_frame=pd.DataFrame(),
        selected_radar=pd.DataFrame(),
        truth=pd.DataFrame(),
        max_eval_time_delta_s=None,
    )

    assert summary["top_residuals"] == []
    assert summary["track_switches"]["posterior_radar"]["count"] == 0
    assert summary["covariance_inflation"]["count"] == 0
    assert summary["worst_time_windows"] == []
