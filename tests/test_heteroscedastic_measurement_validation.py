from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.heteroscedastic_measurements import (
    radar_measurements_to_enu_with_uncertainty,
    rf_measurements_to_enu_with_uncertainty,
)


def test_rf_converter_rejects_invalid_default_std() -> None:
    rf = pd.DataFrame(
        {
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
        }
    )

    for value in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="default_std_m"):
            rf_measurements_to_enu_with_uncertainty(rf, default_std_m=float(value))


def test_radar_converter_rejects_invalid_default_stds() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [2.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
        }
    )

    invalid_cases = (
        {"default_xy_std_m": 0.0},
        {"default_z_std_m": float("nan")},
        {"default_velocity_std_mps": -1.0},
    )
    for kwargs in invalid_cases:
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            radar_measurements_to_enu_with_uncertainty(radar, **kwargs)


def test_valid_default_stds_still_fallback_to_positive_covariance() -> None:
    rf = pd.DataFrame({"time_s": [1.0], "east_m": [10.0], "north_m": [20.0]})
    radar = pd.DataFrame(
        {"time_s": [2.0], "east_m": [10.0], "north_m": [20.0], "up_m": [30.0]}
    )

    [rf_measurement] = rf_measurements_to_enu_with_uncertainty(rf, default_std_m=3.0)
    [radar_measurement] = radar_measurements_to_enu_with_uncertainty(
        radar,
        default_xy_std_m=4.0,
        default_z_std_m=5.0,
    )

    assert np.allclose(rf_measurement.covariance, np.diag([9.0, 9.0]))
    assert np.allclose(radar_measurement.covariance, np.diag([16.0, 16.0, 25.0]))
