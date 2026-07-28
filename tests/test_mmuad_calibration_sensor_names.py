"""Regression tests for MMUAD calibration sensor-name validation."""

from __future__ import annotations

import pytest

from raft_uav.mmuad.calibration import calibration_from_mapping


def _sensor_entry(translation_x_m: float) -> dict[str, object]:
    return {
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "translation_m": [translation_x_m, 0.0, 0.0],
    }


@pytest.mark.parametrize(
    "sensors",
    [
        {
            "radar": _sensor_entry(1.0),
            " RADAR ": _sensor_entry(2.0),
        },
        {
            " RADAR ": _sensor_entry(2.0),
            "radar": _sensor_entry(1.0),
        },
    ],
)
def test_calibration_rejects_ambiguous_normalized_sensor_names(
    sensors: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="calibration sensor names are ambiguous"):
        calibration_from_mapping({"sensors": sensors})


def test_calibration_rejects_blank_sensor_name() -> None:
    with pytest.raises(ValueError, match="calibration sensor names must not be blank"):
        calibration_from_mapping({"sensors": {"  ": _sensor_entry(1.0)}})
