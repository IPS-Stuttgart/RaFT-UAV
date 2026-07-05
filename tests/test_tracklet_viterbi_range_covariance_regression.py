from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from raft_uav.baselines.tracklet_viterbi_range_covariance import _radar_row_covariance


def test_tracklet_range_covariance_ignores_indefinite_row_covariance_for_range_fallback() -> None:
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    row = pd.Series(
        {
            "range_m": 10_000.0,
            "cov_ee": 1.0,
            "cov_nn": 1.0,
            "cov_uu": 1.0,
            "cov_en": 2.0,
            "cov_eu": 0.0,
            "cov_nu": 0.0,
        }
    )
    config = SimpleNamespace(
        use_range_adaptive_radar_covariance=True,
        radar_range_xy_floor_std_m=20.0,
        radar_range_z_floor_std_m=30.0,
        radar_range_xy_scale=0.035,
        radar_range_z_scale=0.050,
    )

    covariance = _radar_row_covariance(row, default_covariance, config)

    np.testing.assert_allclose(covariance, np.diag([350.0**2, 350.0**2, 500.0**2]))
