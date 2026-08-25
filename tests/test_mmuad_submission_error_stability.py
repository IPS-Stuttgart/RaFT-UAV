import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import match_submission_to_truth, metrics_from_matches


def test_matching_keeps_large_representable_euclidean_error_finite() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq-a"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    submission = pd.DataFrame(
        {
            "sequence_id": ["seq-a"],
            "time_s": [0.0],
            "track_id": ["uav0"],
            "x_m": [6.0e307],
            "y_m": [8.0e307],
            "z_m": [0.0],
        }
    )

    matches = match_submission_to_truth(submission, truth, max_time_delta_s=0.0)

    error_2d_m = float(matches.loc[0, "error_2d_m"])
    error_3d_m = float(matches.loc[0, "error_3d_m"])
    assert np.isfinite(error_2d_m)
    assert np.isfinite(error_3d_m)
    assert error_2d_m == pytest.approx(1.0e308, rel=1.0e-12)
    assert error_3d_m == pytest.approx(1.0e308, rel=1.0e-12)


def test_metrics_keep_large_representable_error_statistics_finite() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq-a", "seq-b"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )
    submission = pd.DataFrame(
        {
            "sequence_id": ["seq-a", "seq-b"],
            "time_s": [0.0, 0.0],
            "track_id": ["uav0", "uav0"],
            "x_m": [8.0e307, 1.0e308],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    matches = match_submission_to_truth(submission, truth, max_time_delta_s=0.0)
    metrics = metrics_from_matches(matches, submission=submission, truth=truth)
    pooled = metrics["pooled"]

    expected = {
        "mean_3d_m": 9.0e307,
        "rmse_3d_m": float(np.sqrt(0.82) * 1.0e308),
        "p95_3d_m": 9.9e307,
        "max_3d_m": 1.0e308,
        "fde_3d_m": 9.0e307,
        "mean_2d_m": 9.0e307,
        "p95_2d_m": 9.9e307,
        "max_2d_m": 1.0e308,
        "fde_2d_m": 9.0e307,
    }
    assert pooled["count"] == 2
    for metric, expected_value in expected.items():
        actual = float(pooled[metric])
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected_value, rel=1.0e-12)
