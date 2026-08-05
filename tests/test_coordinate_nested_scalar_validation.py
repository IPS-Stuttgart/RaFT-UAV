from __future__ import annotations

import numpy as np
import pytest

from raft_uav.coordinates import LocalENUProjector


def _boxed(value: object) -> np.ndarray:
    result = np.empty((), dtype=object)
    result[()] = value
    return result


def _cyclic_box() -> np.ndarray:
    result = np.empty((), dtype=object)
    result[()] = result
    return result


@pytest.mark.parametrize(
    "invalid",
    [
        _boxed(np.array(True, dtype=object)),
        _boxed(np.array(np.complex64(48.0 + 2.0j), dtype=object)),
        _boxed(np.array([48.0])),
        _cyclic_box(),
    ],
)
def test_projector_rejects_recursively_boxed_origin_scalars(invalid: object) -> None:
    with pytest.raises(ValueError, match="origin_latitude_deg"):
        LocalENUProjector(invalid, 9.0, 250.0)


@pytest.mark.parametrize(
    "invalid",
    [
        _boxed(np.array(True, dtype=object)),
        _boxed(np.array(np.complex64(48.0 + 2.0j), dtype=object)),
        _boxed(np.array([48.0])),
        _cyclic_box(),
    ],
)
def test_transform_rejects_recursively_boxed_coordinate_scalars(invalid: object) -> None:
    projector = LocalENUProjector(48.0, 9.0, 250.0)

    with pytest.raises(ValueError, match="latitude_deg"):
        projector.transform(invalid, 9.0, 250.0)


@pytest.mark.parametrize(
    "invalid",
    [
        _boxed(np.array(True, dtype=object)),
        _boxed(np.array(np.complex64(48.0 + 2.0j), dtype=object)),
        _boxed(np.array([48.0])),
        _cyclic_box(),
    ],
)
def test_transform_many_rejects_recursively_boxed_coordinate_cells(invalid: object) -> None:
    projector = LocalENUProjector(48.0, 9.0, 250.0)
    latitude = np.empty(2, dtype=object)
    latitude[:] = [48.0, invalid]

    with pytest.raises(ValueError, match="latitude_deg"):
        projector.transform_many(latitude, np.array([9.0, 9.001]), np.array([250.0, 251.0]))


def test_recursively_boxed_real_scalars_remain_supported() -> None:
    latitude = _boxed(_boxed(np.array(48.001)))
    projector = LocalENUProjector(_boxed(np.array(48.0)), 9.0, 250.0)

    actual = projector.transform(latitude, 9.001, 251.0)
    expected = LocalENUProjector(48.0, 9.0, 250.0).transform(48.001, 9.001, 251.0)

    np.testing.assert_allclose(actual, expected)