from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.tracker import add_truth_errors, compute_metrics


def test_tracker_truth_errors_do_not_extrapolate_outside_truth_support() -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, 1.5, 3.0],
            "state_x_m": [100.0, 1.5, 100.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [1.0, 2.0],
            "x_m": [1.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    result = add_truth_errors(estimates, truth)

    assert np.isnan(result.loc[0, "truth_x_m"])
    assert np.isnan(result.loc[0, "error_3d_m"])
    assert result.loc[1, "truth_x_m"] == 1.5
    assert result.loc[1, "error_3d_m"] == 0.0
    assert np.isnan(result.loc[2, "truth_x_m"])
    assert np.isnan(result.loc[2, "error_3d_m"])

    metrics = compute_metrics(result, truth)
    assert metrics["count"] == 1
    assert metrics["mean_3d_m"] == 0.0
