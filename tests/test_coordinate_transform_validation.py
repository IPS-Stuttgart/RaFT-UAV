from __future__ import annotations

import numpy as np
import pytest

from raft_uav.coordinates import LocalENUProjector


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("latitude_deg", np.nan),
        ("latitude_deg", 90.0001),
        ("latitude_deg", True),
        ("longitude_deg", np.inf),
        ("altitude_m", -np.inf),
    ],
)
def test_transform_rejects_invalid_target_coordinates(
    field: str,
    invalid_value: object,
) -> None:
    projector = LocalENUProjector(35.7274895, -78.696216, 2.717)
    coordinates: dict[str, object] = {
        "latitude_deg": 35.7275895,
        "longitude_deg": -78.696116,
        "altitude_m": 30.0,
    }
    coordinates[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        projector.transform(**coordinates)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("latitude_deg", np.array([35.7274895, np.nan])),
        ("latitude_deg", np.array([35.7274895, 91.0])),
        ("latitude_deg", np.array([True, False])),
        ("longitude_deg", np.array([-78.696216, np.inf])),
        ("altitude_m", np.array([2.717, -np.inf])),
    ],
)
def test_transform_many_rejects_invalid_target_coordinates(
    field: str,
    invalid_value: object,
) -> None:
    projector = LocalENUProjector(35.7274895, -78.696216, 2.717)
    coordinates: dict[str, object] = {
        "latitude_deg": np.array([35.7274895, 35.7275895]),
        "longitude_deg": np.array([-78.696216, -78.696116]),
        "altitude_m": 30.0,
    }
    coordinates[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        projector.transform_many(**coordinates)


def test_transform_many_reports_incompatible_coordinate_shapes() -> None:
    projector = LocalENUProjector(35.7274895, -78.696216, 2.717)

    with pytest.raises(ValueError, match="broadcast-compatible"):
        projector.transform_many(
            np.zeros((2, 2)),
            np.zeros(3),
            30.0,
        )
