from __future__ import annotations

import numpy as np

from raft_uav.mmuad.calibration import CalibrationSet, RigidTransform, SensorCalibration


def _make_sensor(source: str) -> SensorCalibration:
    return SensorCalibration(
        source=source,
        transform_sensor_to_world=RigidTransform(
            rotation=np.eye(3),
            translation_m=np.zeros(3),
        ),
    )


def test_calibration_get_does_not_match_specific_key_to_generic_source() -> None:
    calibration = CalibrationSet(sensors={"sensor_detail": _make_sensor("sensor_detail")})

    assert calibration.get("sensor") is None


def test_calibration_get_prefers_longest_forward_prefix() -> None:
    generic = _make_sensor("sensor")
    specific = _make_sensor("sensor_detail")
    calibration = CalibrationSet(sensors={"sensor": generic, "sensor_detail": specific})

    assert calibration.get("sensor_detail_extra") is specific
