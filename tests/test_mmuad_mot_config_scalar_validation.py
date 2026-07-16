from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.mot import MultiObjectTrackerConfig


@pytest.mark.parametrize(
    "field",
    [
        "acceleration_std_mps2",
        "max_association_distance_m",
        "max_track_age_s",
        "min_new_track_confidence",
        "covariance_scale",
    ],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        np.bool_(False),
        np.array(True),
        np.array([1.0]),
        1.0 + 0.0j,
        np.ma.masked,
    ],
)
def test_mot_config_rejects_non_real_scalar_controls(
    field: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        MultiObjectTrackerConfig(**{field: bad_value})


def test_mot_config_accepts_zero_dimensional_real_scalars() -> None:
    config = MultiObjectTrackerConfig(
        acceleration_std_mps2=np.array(8.0),
        max_association_distance_m=np.float64(15.0),
        max_track_age_s=np.array(1.5),
        min_new_track_confidence=np.int64(0),
        covariance_scale=np.array(2.0),
    )

    assert config.acceleration_std_mps2 == 8.0
    assert config.max_association_distance_m == 15.0
    assert config.max_track_age_s == 1.5
    assert config.min_new_track_confidence == 0.0
    assert config.covariance_scale == 2.0
