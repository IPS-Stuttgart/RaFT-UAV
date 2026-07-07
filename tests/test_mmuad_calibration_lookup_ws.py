from __future__ import annotations

import numpy as np

from raft_uav.mmuad.calibration import CalibrationSet, RigidTransform, SensorCalibration


def test_calibration_lookup_ws() -> None:
    sensor = SensorCalibration(
        source="sensor_detail",
        transform_sensor_to_world=RigidTransform(
            rotation=np.eye(3),
            translation_m=np.zeros(3),
        ),
    )
    calibration = CalibrationSet(sensors={" sensor_detail ": sensor})

    assert calibration.get(" sensor_detail_extra ") is sensor
