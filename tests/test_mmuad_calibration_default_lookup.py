import numpy as np

from raft_uav.mmuad.calibration import CalibrationSet, RigidTransform, SensorCalibration


def _sensor(source: str, offset: tuple[float, float, float]) -> SensorCalibration:
    return SensorCalibration(
        source=source,
        transform_sensor_to_world=RigidTransform(
            rotation=np.eye(3),
            translation_m=np.asarray(offset, dtype=float),
        ),
    )


def test_default_calibration_is_fallback_after_specific_lookup() -> None:
    default = _sensor("default", (1.0, 2.0, 3.0))
    specific = _sensor("source_a", (4.0, 5.0, 6.0))
    calibration = CalibrationSet(sensors={"default": default, "source_a": specific})

    assert calibration.get("source_a_extra") is specific
    assert calibration.get("unmatched_source") is default
