from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.io.aerpaw import radar_measurements_to_enu


def _normalized_radar_row() -> dict[str, float]:
    return {
        "time_s": 1.0,
        "east_m": 10.0,
        "north_m": 20.0,
        "up_m": 30.0,
        "velocity_east_mps": 3.0,
        "velocity_north_mps": 4.0,
        "velocity_down_mps": -5.0,
    }


def test_explicit_velocity_mode_rejects_missing_component() -> None:
    row = _normalized_radar_row()
    del row["velocity_down_mps"]

    with pytest.raises(
        ValueError,
        match="requires all radar velocity components",
    ):
        radar_measurements_to_enu(
            pd.DataFrame([row]),
            include_velocity=True,
        )


def test_explicit_velocity_mode_rejects_nonfinite_component() -> None:
    row = _normalized_radar_row()
    row["velocity_north_mps"] = np.nan

    with pytest.raises(
        ValueError,
        match="requires complete finite radar velocity components",
    ):
        radar_measurements_to_enu(
            pd.DataFrame([row]),
            include_velocity=True,
        )


def test_explicit_velocity_mode_keeps_complete_six_dimensional_measurement() -> None:
    [measurement] = radar_measurements_to_enu(
        pd.DataFrame([_normalized_radar_row()]),
        include_velocity=True,
    )

    np.testing.assert_allclose(measurement.vector, [10.0, 20.0, 30.0, 3.0, 4.0, 5.0])
    assert measurement.covariance.shape == (6, 6)
