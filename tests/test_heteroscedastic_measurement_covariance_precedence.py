from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.heteroscedastic_measurements import (
    radar_measurements_to_enu_with_uncertainty,
    rf_measurements_to_enu_with_uncertainty,
)


def test_rf_measurements_prefer_learned_covariance_over_association_covariance() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "cov_ee": [4.0],
            "cov_nn": [9.0],
            "cov_en": [1.0],
            "association_cov_ee": [100.0],
            "association_cov_nn": [121.0],
            "association_cov_en": [0.0],
        }
    )

    [measurement] = rf_measurements_to_enu_with_uncertainty(frame)

    np.testing.assert_allclose(measurement.covariance, [[4.0, 1.0], [1.0, 9.0]])


def test_radar_measurements_prefer_learned_covariance_over_association_covariance() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [2.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "cov_ee": [4.0],
            "cov_nn": [9.0],
            "cov_uu": [16.0],
            "cov_en": [0.0],
            "cov_eu": [0.0],
            "cov_nu": [0.0],
            "association_cov_ee": [100.0],
            "association_cov_nn": [121.0],
            "association_cov_uu": [144.0],
            "association_cov_en": [0.0],
            "association_cov_eu": [0.0],
            "association_cov_nu": [0.0],
        }
    )

    [measurement] = radar_measurements_to_enu_with_uncertainty(frame)

    np.testing.assert_allclose(measurement.covariance, np.diag([4.0, 9.0, 16.0]))
