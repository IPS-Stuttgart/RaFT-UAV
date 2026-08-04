from __future__ import annotations

import numpy as np
import pytest

from raft_uav.coordinates import LocalENUProjector


def _nested_zero_dimensional(value: object, *, depth: int = 2) -> np.ndarray:
    nested = value
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = nested
        nested = wrapper
    assert isinstance(nested, np.ndarray)
    return nested


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("origin_latitude_deg", _nested_zero_dimensional(np.bool_(True), depth=3)),
        ("origin_longitude_deg", _nested_zero_dimensional(np.array([-78.0]))),
        ("origin_altitude_m", _nested_zero_dimensional(np.array([10.0]))),
    ],
)
def test_projector_rejects_nested_non_scalar_origin_values(
    field: str,
    invalid_value: object,
) -> None:
    coordinates: dict[str, object] = {
        "origin_latitude_deg": 35.0,
        "origin_longitude_deg": -78.0,
        "origin_altitude_m": 10.0,
    }
    coordinates[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        LocalENUProjector(**coordinates)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("latitude_deg", _nested_zero_dimensional(np.bool_(True), depth=3)),
        ("longitude_deg", _nested_zero_dimensional(np.array([-78.0]))),
        ("altitude_m", _nested_zero_dimensional(np.array([10.0]))),
    ],
)
def test_transform_rejects_nested_non_scalar_coordinate_values(
    field: str,
    invalid_value: object,
) -> None:
    projector = LocalENUProjector(35.0, -78.0, 10.0)
    coordinates: dict[str, object] = {
        "latitude_deg": 35.1,
        "longitude_deg": -78.1,
        "altitude_m": 11.0,
    }
    coordinates[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        projector.transform(**coordinates)


@pytest.mark.parametrize(
    ("field", "hidden_value"),
    [
        ("latitude_deg", np.bool_(True)),
        ("longitude_deg", np.array([-78.1])),
        ("altitude_m", np.array([11.0])),
    ],
)
def test_transform_many_rejects_nested_object_array_cells(
    field: str,
    hidden_value: object,
) -> None:
    projector = LocalENUProjector(35.0, -78.0, 10.0)
    coordinates: dict[str, object] = {
        "latitude_deg": np.array([35.0, 35.1]),
        "longitude_deg": np.array([-78.0, -78.1]),
        "altitude_m": np.array([10.0, 11.0]),
    }
    invalid = np.empty(2, dtype=object)
    invalid[0] = coordinates[field][0]
    invalid[1] = _nested_zero_dimensional(hidden_value, depth=3)
    coordinates[field] = invalid

    with pytest.raises(ValueError, match=field):
        projector.transform_many(**coordinates)


def test_projector_accepts_recursively_nested_real_scalars() -> None:
    projector = LocalENUProjector(
        _nested_zero_dimensional(35.0, depth=3),
        _nested_zero_dimensional(-78.0, depth=3),
        _nested_zero_dimensional(10.0, depth=3),
    )
    latitude = np.empty(2, dtype=object)
    latitude[0] = _nested_zero_dimensional(35.0, depth=2)
    latitude[1] = _nested_zero_dimensional(35.1, depth=3)

    transformed = projector.transform_many(
        latitude,
        np.array([-78.0, -78.1]),
        10.0,
    )

    assert transformed.shape == (2, 3)


def test_projector_rejects_cyclic_scalar_arrays() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match="origin_latitude_deg"):
        LocalENUProjector(cyclic, -78.0, 10.0)
