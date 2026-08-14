import numpy as np
import pytest

from raft_uav.coordinates import LocalENUProjector


def _boxed(value: object) -> np.ndarray:
    array = np.empty(1, dtype=object)
    array[0] = value
    return array


@pytest.mark.parametrize(
    ("field", "error_name"),
    [
        ("latitude", "latitude_deg"),
        ("longitude", "longitude_deg"),
        ("altitude", "altitude_m"),
    ],
)
def test_transform_rejects_boolean_hidden_in_zero_dim_array(
    field: str,
    error_name: str,
) -> None:
    projector = LocalENUProjector(0.0, 0.0, 0.0)
    coordinates: dict[str, object] = {
        "latitude": 0.0,
        "longitude": 0.0,
        "altitude": 0.0,
    }
    coordinates[field] = np.array(True)

    with pytest.raises(ValueError, match=rf"{error_name} must be a finite real scalar"):
        projector.transform(
            coordinates["latitude"],
            coordinates["longitude"],
            coordinates["altitude"],
        )


def test_projector_origin_rejects_boolean_hidden_in_zero_dim_array() -> None:
    with pytest.raises(
        ValueError,
        match=r"origin_latitude_deg must be a finite real scalar",
    ):
        LocalENUProjector(np.array(True), 0.0, 0.0)


@pytest.mark.parametrize(
    ("field", "error_name"),
    [
        ("latitude", "latitude_deg"),
        ("longitude", "longitude_deg"),
        ("altitude", "altitude_m"),
    ],
)
def test_transform_many_rejects_boolean_hidden_in_zero_dim_object_array(
    field: str,
    error_name: str,
) -> None:
    projector = LocalENUProjector(0.0, 0.0, 0.0)
    coordinates = {
        "latitude": np.array([0.0]),
        "longitude": np.array([0.0]),
        "altitude": np.array([0.0]),
    }
    coordinates[field] = _boxed(np.array(True))

    with pytest.raises(ValueError, match=rf"{error_name} must contain finite real values"):
        projector.transform_many(
            coordinates["latitude"],
            coordinates["longitude"],
            coordinates["altitude"],
        )


def test_transform_many_rejects_recursively_boxed_boolean() -> None:
    projector = LocalENUProjector(0.0, 0.0, 0.0)
    inner = np.empty((), dtype=object)
    inner[()] = np.array(True)

    with pytest.raises(ValueError, match=r"latitude_deg must contain finite real values"):
        projector.transform_many(_boxed(inner), np.array([0.0]), np.array([0.0]))


def test_transform_many_rejects_nested_non_scalar_array() -> None:
    projector = LocalENUProjector(0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match=r"latitude_deg must contain finite real values"):
        projector.transform_many(_boxed(np.array([1.0])), np.array([0.0]), np.array([0.0]))


def test_transform_many_rejects_cyclic_object_scalar() -> None:
    projector = LocalENUProjector(0.0, 0.0, 0.0)
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match=r"latitude_deg must contain finite real values"):
        projector.transform_many(_boxed(cyclic), np.array([0.0]), np.array([0.0]))


def test_transform_accepts_boxed_zero_dim_real_scalar() -> None:
    projector = LocalENUProjector(0.0, 0.0, 0.0)
    expected = projector.transform(1.25, 2.5, 3.75)
    actual = projector.transform(np.array(1.25), np.array(2.5), np.array(3.75))

    np.testing.assert_allclose(actual, expected)


def test_transform_many_accepts_boxed_zero_dim_real_scalar() -> None:
    projector = LocalENUProjector(0.0, 0.0, 0.0)
    expected = projector.transform_many(
        np.array([1.25]),
        np.array([2.5]),
        np.array([3.75]),
    )
    actual = projector.transform_many(
        _boxed(np.array(1.25)),
        _boxed(np.array(2.5)),
        _boxed(np.array(3.75)),
    )

    np.testing.assert_allclose(actual, expected)
