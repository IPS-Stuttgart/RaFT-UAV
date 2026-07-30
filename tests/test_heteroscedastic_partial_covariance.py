from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.heteroscedastic_measurements import (
    radar_measurements_to_enu_with_uncertainty,
    rf_measurements_to_enu_with_uncertainty,
)


def test_rf_partial_learned_covariance_preserves_available_axis() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "std_m": [20.0],
            "cov_ee": [4.0],
            "cov_en": [0.0],
        }
    )

    [measurement] = rf_measurements_to_enu_with_uncertainty(frame)

    np.testing.assert_allclose(
        measurement.covariance,
        np.diag([4.0, 400.0]),
    )


def test_radar_partial_learned_covariance_uses_association_fallback() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [2.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "cov_ee": [4.0],
            "cov_nn": [9.0],
            "cov_en": [0.0],
            "cov_eu": [0.0],
            "cov_nu": [0.0],
            "association_cov_ee": [100.0],
            "association_cov_nn": [121.0],
            "association_cov_uu": [144.0],
            "association_cov_en": [1.0],
            "association_cov_eu": [2.0],
            "association_cov_nu": [3.0],
        }
    )

    [measurement] = radar_measurements_to_enu_with_uncertainty(frame)

    np.testing.assert_allclose(
        measurement.covariance,
        np.array(
            [
                [4.0, 0.0, 2.0],
                [0.0, 9.0, 3.0],
                [2.0, 3.0, 144.0],
            ]
        ),
    )
