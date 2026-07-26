from __future__ import annotations

import numpy as np
import pytest

from raft_uav.coordinates import LocalENUProjector


@pytest.mark.parametrize(
    ("field", "real_part"),
    [
        ("origin_latitude_deg", 48.0),
        ("origin_longitude_deg", 9.0),
        ("origin_altitude_m", 250.0),
    ],
)
def test_projector_rejects_object_wrapped_numpy_complex_origin(
    field: str,
    real_part: float,
) -> None:
    origin: dict[str, object] = {
        "origin_latitude_deg": 48.0,
        "origin_longitude_deg": 9.0,
        "origin_altitude_m": 250.0,
    }
    origin[field] = np.array(np.complex64(real_part + 1.0j), dtype=object)

    with pytest.raises(ValueError, match=field):
        LocalENUProjector(**origin)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        (
            "latitude_deg",
            np.array([np.float64(48.0), np.complex64(48.001 + 1.0j)], dtype=object),
        ),
        (
            "longitude_deg",
            np.array([np.float64(9.0), np.complex64(9.001 + 1.0j)], dtype=object),
        ),
        (
            "altitude_m",
            np.array([np.float64(250.0), np.complex64(260.0 + 1.0j)], dtype=object),
        ),
    ],
)
def test_transform_many_rejects_object_arrays_with_numpy_complex_scalars(
    field: str,
    values: np.ndarray,
) -> None:
    projector = LocalENUProjector(48.0, 9.0, 250.0)
    coordinates: dict[str, object] = {
        "latitude_deg": np.array([48.0, 48.001]),
        "longitude_deg": np.array([9.0, 9.001]),
        "altitude_m": np.array([250.0, 260.0]),
    }
    coordinates[field] = values

    with pytest.raises(ValueError, match=field):
        projector.transform_many(**coordinates)
