import numpy as np
import pytest

from raft_uav.evaluation.metrics import summarize_errors


def test_summarize_errors_rejects_negative_position_errors():
    with pytest.raises(ValueError, match="non-negative"):
        summarize_errors(np.array([1.0, -0.25, np.nan]))


def test_summarize_errors_keeps_large_finite_statistics_finite():
    summary = summarize_errors(np.array([8.0e307, 1.0e308]))
    expected = {
        "mean_m": 9.0e307,
        "std_m": 1.0e307,
        "rmse_m": float(np.sqrt(0.82) * 1.0e308),
        "mae_m": 9.0e307,
        "p50_m": 9.0e307,
        "p95_m": 9.9e307,
        "max_m": 1.0e308,
    }

    assert summary["count"] == 2.0
    for metric, expected_value in expected.items():
        actual = float(summary[metric])
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected_value, rel=1.0e-12)
