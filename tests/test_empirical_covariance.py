import numpy as np
import pandas as pd

from raft_uav.calibration.empirical_covariance import (
    apply_empirical_covariance,
    estimate_empirical_measurement_covariances,
)
from raft_uav.uncertainty import covariance_from_row


def test_empirical_covariance_estimates_and_writes_cov_columns():
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 10.0, 20.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )
    rf = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [1.0, 11.0, 19.0],
            "north_m": [2.0, -2.0, 0.0],
        }
    )
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [2.0, 8.0, 20.0],
            "north_m": [1.0, -1.0, 0.0],
            "up_m": [3.0, 0.0, -3.0],
        }
    )

    payload = estimate_empirical_measurement_covariances(
        rf=rf,
        radar=radar,
        truth=truth,
        max_time_delta_s=0.25,
        min_variance_m2=0.5,
    )
    assert payload["rf"]["sample_count"] == 3
    assert payload["radar"]["sample_count"] == 3

    rf_with_cov = apply_empirical_covariance(rf, source="rf", covariance_payload=payload)
    radar_with_cov = apply_empirical_covariance(radar, source="radar", covariance_payload=payload)

    assert {"cov_ee", "cov_nn", "cov_en"}.issubset(rf_with_cov.columns)
    assert {"cov_ee", "cov_nn", "cov_uu", "cov_en", "cov_eu", "cov_nu"}.issubset(
        radar_with_cov.columns
    )
    rf_cov = covariance_from_row(rf_with_cov.iloc[0], 2, np.eye(2))
    radar_cov = covariance_from_row(radar_with_cov.iloc[0], 3, np.eye(3))
    assert rf_cov.shape == (2, 2)
    assert radar_cov.shape == (3, 3)
    assert np.all(np.diag(rf_cov) > 0.0)
    assert np.all(np.diag(radar_cov) > 0.0)


def test_empirical_covariance_respects_time_gate():
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    rf = pd.DataFrame({"time_s": [10.0], "east_m": [1.0], "north_m": [1.0]})

    payload = estimate_empirical_measurement_covariances(
        rf=rf,
        radar=None,
        truth=truth,
        max_time_delta_s=0.5,
    )

    assert "rf" not in payload
