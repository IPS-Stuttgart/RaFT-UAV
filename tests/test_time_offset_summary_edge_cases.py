from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.time_offset import summarize_errors, truth_position_at_time


def test_truth_position_at_time_returns_none_when_no_truth_sample_is_close() -> None:
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 10.0],
        }
    )

    position = truth_position_at_time(truth, 5.0, max_delta_s=1.0)

    assert position is None


def test_summarize_errors_returns_nan_metrics_for_nonfinite_errors() -> None:
    summary = summarize_errors(
        tau_s=0.5,
        candidate_count=2,
        selected_count=1,
        matched_count=0,
        errors_m=np.array([np.nan, np.inf]),
    )

    assert summary["coverage"] == 0.0
    assert summary["selected_coverage"] == 0.0
    for metric in (
        "mean_error_m",
        "std_error_m",
        "rmse_error_m",
        "p50_error_m",
        "p95_error_m",
        "max_error_m",
    ):
        assert np.isnan(float(summary[metric]))


def test_summarize_errors_keeps_large_finite_statistics_finite() -> None:
    summary = summarize_errors(
        tau_s=0.5,
        candidate_count=2,
        selected_count=2,
        matched_count=2,
        errors_m=np.array([8.0e307, 1.0e308]),
    )
    expected = {
        "mean_error_m": 9.0e307,
        "std_error_m": 1.0e307,
        "rmse_error_m": float(np.sqrt(0.82) * 1.0e308),
        "p50_error_m": 9.0e307,
        "p95_error_m": 9.9e307,
        "max_error_m": 1.0e308,
    }

    assert summary["coverage"] == 1.0
    assert summary["selected_coverage"] == 1.0
    for metric, expected_value in expected.items():
        actual = float(summary[metric])
        assert np.isfinite(actual)
        assert np.isclose(actual, expected_value, rtol=1.0e-12)
