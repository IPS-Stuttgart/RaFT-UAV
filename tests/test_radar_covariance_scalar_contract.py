import numpy as np
import pytest

from raft_uav.baselines.radar_covariance import (
    RadarCovarianceConfig,
    fixed_radar_covariance,
)


_NUMERIC_CONFIG_FIELDS = (
    "xy_std_m",
    "z_std_m",
    "range_std_m",
    "azimuth_std_deg",
    "elevation_std_deg",
    "min_std_m",
    "max_std_m",
    "origin_east_m",
    "origin_north_m",
    "origin_up_m",
)


@pytest.mark.parametrize("field", _NUMERIC_CONFIG_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        np.complex128(5.0 + 2.0j),
        np.array(np.complex128(5.0 + 2.0j), dtype=object),
    ],
)
def test_radar_covariance_config_rejects_lossy_scalar_controls(field, value):
    with pytest.raises(ValueError, match=field):
        RadarCovarianceConfig(**{field: value})


@pytest.mark.parametrize(
    ("xy_std_m", "z_std_m", "field"),
    [
        (True, 35.0, "xy_std_m"),
        (25.0, np.complex128(35.0 + 1.0j), "z_std_m"),
    ],
)
def test_fixed_radar_covariance_rejects_lossy_scalar_controls(
    xy_std_m,
    z_std_m,
    field,
):
    with pytest.raises(ValueError, match=field):
        fixed_radar_covariance(xy_std_m, z_std_m)


def test_radar_covariance_config_normalizes_valid_scalar_like_values():
    config = RadarCovarianceConfig(
        xy_std_m=np.array(12.0, dtype=np.float32),
        origin_east_m="3.5",
    )

    assert type(config.xy_std_m) is float
    assert type(config.origin_east_m) is float
    assert config.xy_std_m == 12.0
    assert config.origin_east_m == 3.5
    np.testing.assert_allclose(config.fixed_covariance(), np.diag([144.0, 144.0, 1225.0]))
