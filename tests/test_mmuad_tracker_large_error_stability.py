from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.tracker import add_truth_errors, compute_metrics


def test_truth_error_norm_keeps_large_finite_diagonal_representable() -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [0.0],
            "state_x_m": [1.0e308],
            "state_y_m": [1.0e308],
            "state_z_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        out = add_truth_errors(estimates, truth)

    expected = float(np.hypot(1.0e308, 1.0e308))
    assert np.isfinite(out.loc[0, "error_2d_m"])
    assert np.isfinite(out.loc[0, "error_3d_m"])
    assert out.loc[0, "error_2d_m"] == pytest.approx(expected)
    assert out.loc[0, "error_3d_m"] == pytest.approx(expected)


def test_tracker_metrics_keep_large_finite_mean_and_rmse_representable() -> None:
    estimates = pd.DataFrame(
        {
            "error_3d_m": [1.0e200, 1.0e200],
            "error_2d_m": [1.0e200, 1.0e200],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        metrics = compute_metrics(estimates, truth=None)

    assert metrics["count"] == 2
    assert metrics["mean_3d_m"] == pytest.approx(1.0e200)
    assert metrics["rmse_3d_m"] == pytest.approx(1.0e200)
    assert metrics["mean_2d_m"] == pytest.approx(1.0e200)
