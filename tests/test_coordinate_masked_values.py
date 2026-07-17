from __future__ import annotations

import numpy as np
import pytest

from raft_uav.coordinates import LocalENUProjector


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("origin_latitude_deg", np.ma.array(48.0, mask=True)),
        ("origin_longitude_deg", np.ma.array(9.0, mask=True)),
        ("origin_altitude_m", np.ma.array(250.0, mask=True)),
    ],
)
def test_projector_rejects_masked_origin_coordinates(
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


@pytest.mark.parametrize(
    "field",
    ["latitude_deg", "longitude_deg", "altitude_m"],
)
def test_transform_rejects_masked_target_coordinates(field: str) -> None:
    projector = LocalENUProjector(48.0, 9.0, 250.0)
    coordinates: dict[str, object] = {
        "latitude_deg": 48.001,
        "longitude_deg": 9.001,
        "altitude_m": 260.0,
    }
    coordinates[field] = np.ma.masked

    with pytest.raises(ValueError, match=field):
        projector.transform(**coordinates)


@pytest.mark.parametrize(
    "field",
    ["latitude_deg", "longitude_deg", "altitude_m"],
)
def test_transform_many_rejects_partially_masked_coordinates(field: str) -> None:
    projector = LocalENUProjector(48.0, 9.0, 250.0)
    coordinates: dict[str, object] = {
        "latitude_deg": np.array([48.0, 48.001]),
        "longitude_deg": np.array([9.0, 9.001]),
        "altitude_m": np.array([250.0, 260.0]),
    }
    coordinates[field] = np.ma.array(coordinates[field], mask=[False, True])

    with pytest.raises(ValueError, match=field):
        projector.transform_many(**coordinates)


def test_transform_many_accepts_unmasked_masked_arrays() -> None:
    projector = LocalENUProjector(48.0, 9.0, 250.0)

    result = projector.transform_many(
        np.ma.array([48.0, 48.001], mask=False),
        np.ma.array([9.0, 9.001], mask=False),
        np.ma.array([250.0, 260.0], mask=False),
    )

    assert result.shape == (2, 3)
    assert np.isfinite(result).all()
