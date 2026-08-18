from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.heteroscedastic_measurements import (
    radar_measurements_to_enu_with_uncertainty,
)


def _radar_frame(**velocity_columns: list[object]) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "time_s": [2.0],
        "east_m": [10.0],
        "north_m": [20.0],
        "up_m": [30.0],
        "cov_ee": [16.0],
        "cov_nn": [25.0],
        "cov_uu": [36.0],
    }
    data.update(velocity_columns)
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "radar",
    [
        _radar_frame(
            velocity_east_mps=[1.0],
            velocity_north_mps=[None],
            velocity_down_mps=[-3.0],
        ),
        _radar_frame(
            velocity_east_mps=[1.0],
            velocity_north_mps=[2.0],
        ),
        _radar_frame(
            velocity_east_mps=[1.0],
            velocity_north_mps=[2.0],
            velocity_down_mps=[np.inf],
        ),
    ],
)
def test_heteroscedastic_radar_converter_rejects_partial_velocity(
    radar: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="radar velocity components must be all missing or a complete finite",
    ):
        radar_measurements_to_enu_with_uncertainty(radar)


def test_heteroscedastic_radar_converter_keeps_all_null_velocity_fallback() -> None:
    radar = _radar_frame(
        velocity_east_mps=[None],
        velocity_north_mps=[np.nan],
        velocity_down_mps=[pd.NA],
    )

    [measurement] = radar_measurements_to_enu_with_uncertainty(radar)

    assert measurement.vector.shape == (3,)
    np.testing.assert_allclose(measurement.vector, [10.0, 20.0, 30.0])
