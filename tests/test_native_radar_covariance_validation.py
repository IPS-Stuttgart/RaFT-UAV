import numpy as np
import pytest

from raft_uav.research.measurement_models import enu_covariance_from_range_az_el


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"range_m": -1.0}, "range_m"),
        ({"azimuth_rad": np.nan}, "azimuth_rad"),
        ({"elevation_rad": np.inf}, "elevation_rad"),
        ({"range_std_m": -1.0}, "range_std_m"),
        ({"azimuth_std_rad": np.inf}, "azimuth_std_rad"),
        ({"elevation_std_rad": np.nan}, "elevation_std_rad"),
        ({"min_std_m": -0.1}, "min_std_m"),
    ],
)
def test_native_covariance_rejects_invalid_geometry_or_uncertainty(kwargs, message):
    params = {
        "range_m": 100.0,
        "azimuth_rad": 0.5,
        "elevation_rad": 0.1,
        "range_std_m": 5.0,
        "azimuth_std_rad": 0.02,
        "elevation_std_rad": 0.03,
        "min_std_m": 1.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        enu_covariance_from_range_az_el(**params)
