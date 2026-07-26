from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.io.aerpaw import radar_measurements_to_enu


def _normalized_radar(*, with_velocity: bool = False) -> pd.DataFrame:
    data: dict[str, list[float]] = {
        "time_s": [1.0],
        "east_m": [10.0],
        "north_m": [20.0],
        "up_m": [30.0],
    }
    if with_velocity:
        data.update(
            {
                "velocity_east_mps": [2.0],
                "velocity_north_mps": [3.0],
                "velocity_down_mps": [4.0],
            }
        )
    return pd.DataFrame(data)


def test_radar_converter_uses_validated_position_covariance_defaults() -> None:
    [measurement] = radar_measurements_to_enu(
        _normalized_radar(),
        default_xy_std_m=4.0,
        default_z_std_m=5.0,
    )

    np.testing.assert_allclose(measurement.covariance, np.diag([16.0, 16.0, 25.0]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_xy_std_m", None),
        ("default_xy_std_m", True),
        ("default_xy_std_m", 0.0),
        ("default_xy_std_m", -1.0),
        ("default_xy_std_m", np.nan),
        ("default_xy_std_m", np.inf),
        ("default_xy_std_m", 1.0 + 2.0j),
        ("default_xy_std_m", np.array([1.0])),
        ("default_z_std_m", None),
        ("default_z_std_m", True),
        ("default_z_std_m", 0.0),
        ("default_z_std_m", -1.0),
        ("default_z_std_m", np.nan),
        ("default_z_std_m", np.inf),
        ("default_z_std_m", 1.0 + 2.0j),
        ("default_z_std_m", np.array([1.0])),
    ],
)
def test_radar_converter_rejects_invalid_position_covariance_defaults(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        radar_measurements_to_enu(_normalized_radar(), **{field: value})


@pytest.mark.parametrize("value", [None, True, 0.0, -1.0, np.nan, np.inf, 1.0 + 2.0j])
def test_radar_converter_rejects_invalid_velocity_std_when_enabled(value: object) -> None:
    with pytest.raises(ValueError, match="default_velocity_std_mps"):
        radar_measurements_to_enu(
            _normalized_radar(with_velocity=True),
            include_velocity=True,
            default_velocity_std_mps=value,
        )


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", np.array([True])])
def test_radar_converter_rejects_nonboolean_velocity_control(value: object) -> None:
    with pytest.raises(ValueError, match="include_velocity"):
        radar_measurements_to_enu(_normalized_radar(), include_velocity=value)


def test_radar_converter_accepts_numpy_boolean_velocity_control() -> None:
    [measurement] = radar_measurements_to_enu(
        _normalized_radar(with_velocity=True),
        default_xy_std_m=4.0,
        default_z_std_m=5.0,
        default_velocity_std_mps=6.0,
        include_velocity=np.bool_(True),
    )

    np.testing.assert_allclose(
        measurement.vector,
        np.array([10.0, 20.0, 30.0, 2.0, 3.0, -4.0]),
    )
    np.testing.assert_allclose(
        measurement.covariance,
        np.diag([16.0, 16.0, 25.0, 36.0, 36.0, 36.0]),
    )
