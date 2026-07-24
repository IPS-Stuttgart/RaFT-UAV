from __future__ import annotations

import numpy as np
import pytest

from raft_uav.coordinates import LocalENUProjector


@pytest.mark.parametrize("longitude_deg", [-180.0, 180.0])
def test_projector_accepts_longitude_interval_endpoints(longitude_deg: float) -> None:
    projector = LocalENUProjector(48.0, longitude_deg, 250.0)

    result = projector.transform(48.001, longitude_deg, 260.0)

    assert np.isfinite(result).all()


@pytest.mark.parametrize("longitude_deg", [-180.0001, 180.0001])
def test_projector_rejects_out_of_range_origin_longitude(longitude_deg: float) -> None:
    with pytest.raises(ValueError, match="origin_longitude_deg"):
        LocalENUProjector(48.0, longitude_deg, 250.0)


@pytest.mark.parametrize("longitude_deg", [-180.0001, 180.0001])
def test_transform_rejects_out_of_range_longitude(longitude_deg: float) -> None:
    projector = LocalENUProjector(48.0, 9.0, 250.0)

    with pytest.raises(ValueError, match="longitude_deg"):
        projector.transform(48.001, longitude_deg, 260.0)


def test_transform_many_rejects_any_out_of_range_longitude() -> None:
    projector = LocalENUProjector(48.0, 9.0, 250.0)

    with pytest.raises(ValueError, match="longitude_deg"):
        projector.transform_many(
            np.array([48.0, 48.001]),
            np.array([9.0, 181.0]),
            np.array([250.0, 260.0]),
        )
