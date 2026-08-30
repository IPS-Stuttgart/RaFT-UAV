"""Extended NumPy scalars must not recurse indefinitely during validation."""

import numpy as np
import pytest

from raft_uav.coordinates import LocalENUProjector


ORIGIN = (35.727, -78.695, 70.0)
POINT = (35.728, -78.694, 75.0)
COORDINATE_NAMES = ("latitude_deg", "longitude_deg", "altitude_m")


def _object_scalar(value):
    result = np.empty((), dtype=object)
    result[()] = value
    return result


def _wrap(value, container):
    if container == "array":
        return np.asarray(value)
    if container == "object":
        return _object_scalar(value)
    if container == "nested-object":
        return _object_scalar(_object_scalar(value))
    return value


@pytest.mark.parametrize("coordinate", range(3), ids=COORDINATE_NAMES)
@pytest.mark.parametrize("container", ["scalar", "array", "object", "nested-object"])
def test_longdouble_origin_matches_float_origin(coordinate, container):
    origin = list(ORIGIN)
    origin[coordinate] = _wrap(np.longdouble(origin[coordinate]), container)

    actual = LocalENUProjector(*origin)
    reference = LocalENUProjector(*ORIGIN)

    assert isinstance(getattr(actual, f"origin_{COORDINATE_NAMES[coordinate]}"), float)
    np.testing.assert_allclose(
        actual.transform(*POINT), reference.transform(*POINT), rtol=0.0, atol=1e-9
    )


@pytest.mark.parametrize("coordinate", range(3), ids=COORDINATE_NAMES)
@pytest.mark.parametrize("container", ["scalar", "array", "object", "nested-object"])
def test_longdouble_transform_matches_float_transform(coordinate, container):
    projector = LocalENUProjector(*ORIGIN)
    point = list(POINT)
    point[coordinate] = _wrap(np.longdouble(point[coordinate]), container)

    np.testing.assert_allclose(
        projector.transform(*point), projector.transform(*POINT), rtol=0.0, atol=1e-9
    )


@pytest.mark.parametrize("coordinate", range(3), ids=COORDINATE_NAMES)
@pytest.mark.parametrize("container", ["scalar", "array", "object", "nested-object"])
def test_longdouble_object_batch_matches_float_batch(coordinate, container):
    projector = LocalENUProjector(*ORIGIN)
    point = list(POINT)
    values = np.empty(2, dtype=object)
    values[0] = _wrap(np.longdouble(POINT[coordinate]), container)
    values[1] = POINT[coordinate] + 0.001
    point[coordinate] = values
    reference_point = list(POINT)
    reference_point[coordinate] = np.array([POINT[coordinate], values[1]])

    np.testing.assert_allclose(
        projector.transform_many(*point),
        projector.transform_many(*reference_point),
        rtol=0.0,
        atol=1e-9,
    )


@pytest.mark.parametrize("coordinate", range(3), ids=COORDINATE_NAMES)
@pytest.mark.parametrize("operation", ["origin", "transform", "transform_many"])
@pytest.mark.parametrize(
    "value",
    [
        np.longdouble("nan"),
        np.longdouble("inf"),
        np.longdouble("-inf"),
        np.clongdouble(1.0 + 0.0j),
        np.clongdouble(1.0 + 2.0j),
    ],
    ids=["nan", "inf", "negative-inf", "complex-zero-imag", "complex-nonzero-imag"],
)
def test_invalid_extended_scalars_raise_field_specific_value_error(coordinate, operation, value):
    point = list(ORIGIN)
    point[coordinate] = _object_scalar(value)
    name = COORDINATE_NAMES[coordinate]
    if operation == "origin":
        call = LocalENUProjector
        error = f"origin_{name} must be a finite real scalar"
    else:
        call = getattr(LocalENUProjector(*ORIGIN), operation)
        error = (
            f"{name} must contain finite real values"
            if operation == "transform_many"
            else f"{name} must be a finite real scalar"
        )

    with pytest.raises(ValueError, match=f"^{error}$"):
        call(*point)


@pytest.mark.parametrize("operation", ["origin", "transform", "transform_many"])
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(False), np.ma.masked, [1.0], np.array([1.0]), 1.0 + 0.0j],
    ids=["bool", "numpy-bool", "masked", "list", "non-scalar-array", "complex"],
)
def test_object_wrapping_does_not_bypass_other_scalar_validation(operation, value):
    point = list(ORIGIN)
    point[0] = _object_scalar(value)
    call = (
        LocalENUProjector
        if operation == "origin"
        else getattr(LocalENUProjector(*ORIGIN), operation)
    )

    with pytest.raises(ValueError, match="latitude_deg must"):
        call(*point)


@pytest.mark.parametrize("operation", ["origin", "transform", "transform_many"])
def test_self_referential_object_scalar_is_still_rejected(operation):
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    call = (
        LocalENUProjector
        if operation == "origin"
        else getattr(LocalENUProjector(*ORIGIN), operation)
    )

    with pytest.raises(ValueError, match="latitude_deg must"):
        call(cyclic, ORIGIN[1], ORIGIN[2])
