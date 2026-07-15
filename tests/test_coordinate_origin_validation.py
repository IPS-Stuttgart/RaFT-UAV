from __future__ import annotations

import numpy as np
import pytest

from raft_uav.coordinates import LocalENUProjector


@pytest.mark.parametrize(
    "field",
    [
        "origin_latitude_deg",
        "origin_longitude_deg",
        "origin_altitude_m",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.array([0.0]),
        "invalid",
        1.0 + 2.0j,
    ],
)
def test_projector_rejects_invalid_origin_scalars(
    field: str,
    value: object,
) -> None:
    origin: dict[str, object] = {
        "origin_latitude_deg": 48.0,
        "origin_longitude_deg": 9.0,
        "origin_altitude_m": 250.0,
    }
    origin[field] = value

    with pytest.raises(ValueError, match=field):
        LocalENUProjector(**origin)


@pytest.mark.parametrize("latitude_deg", [-90.000001, 90.000001])
def test_projector_rejects_latitudes_outside_wgs84_range(
    latitude_deg: float,
) -> None:
    with pytest.raises(ValueError, match="between -90 and 90"):
        LocalENUProjector(latitude_deg, 9.0, 250.0)


def test_projector_normalizes_valid_scalar_like_origins() -> None:
    projector = LocalENUProjector(
        origin_latitude_deg=np.array(48.0),
        origin_longitude_deg="9.0",
        origin_altitude_m=np.float64(250.0),
    )

    assert projector.origin_latitude_deg == 48.0
    assert projector.origin_longitude_deg == 9.0
    assert projector.origin_altitude_m == 250.0
    np.testing.assert_allclose(
        projector.transform(48.0, 9.0, 250.0),
        np.zeros(3),
        atol=1.0e-8,
    )
