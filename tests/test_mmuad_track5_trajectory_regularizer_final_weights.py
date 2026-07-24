from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_regularizer import regularize_track5_estimates


def test_regularizer_reports_weights_for_final_smoothed_state() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 100.0, 3.0, 4.0],
            "state_y_m": [0.0] * 5,
            "state_z_m": [1.0] * 5,
        }
    )

    regularized, diagnostics = regularize_track5_estimates(
        estimates,
        smoothness_weight=100.0,
        huber_delta_m=10.0,
        iterations=1,
        observation_sigma_m=1.0,
    )

    residual = regularized["regularizer_residual_m"].to_numpy(dtype=float)
    reported = regularized["regularizer_robust_weight"].to_numpy(dtype=float)
    expected = np.minimum(1.0, 10.0 / np.maximum(residual, 1.0e-12))

    np.testing.assert_allclose(reported, expected)
    outlier = regularized.loc[regularized["time_s"] == 2.0].iloc[0]
    assert outlier["regularizer_robust_weight"] < 1.0
    assert diagnostics.loc[0, "mean_robust_weight"] == pytest.approx(expected.mean())
