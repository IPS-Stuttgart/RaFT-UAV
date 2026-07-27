"""Regression tests for complex-valued MMUAD calibration inputs."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from raft_uav.mmuad.calibration import RigidTransform, calibration_from_mapping

_COMPLEX_WARNING = getattr(getattr(np, "exceptions", np), "ComplexWarning")


def test_rigid_transform_rejects_complex_rotation_before_cast() -> None:
    rotation = np.eye(3, dtype=np.complex128)
    rotation[0, 0] += 2.0j

    with warnings.catch_warnings():
        warnings.simplefilter("error", _COMPLEX_WARNING)
        with pytest.raises(ValueError, match="rotation must contain real values"):
            RigidTransform(rotation=rotation, translation_m=np.zeros(3))


def test_rigid_transform_rejects_complex_translation_before_cast() -> None:
    translation = np.array([0.0, np.complex64(1.0 + 2.0j), 0.0], dtype=object)

    with warnings.catch_warnings():
        warnings.simplefilter("error", _COMPLEX_WARNING)
        with pytest.raises(ValueError, match="translation_m must contain real values"):
            RigidTransform(rotation=np.eye(3), translation_m=translation)


def test_mapping_rejects_object_wrapped_complex_rotation() -> None:
    payload = {
        "sensors": {
            "radar": {
                "rotation_matrix": np.array(
                    [
                        [np.complex64(1.0 + 1.0j), 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=object,
                )
            }
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", _COMPLEX_WARNING)
        with pytest.raises(ValueError, match="calibration matrix must contain real values"):
            calibration_from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("quaternion_wxyz", [np.complex64(1.0 + 1.0j), 0.0, 0.0, 0.0], "quaternion"),
        ("rpy_deg", [0.0, np.complex64(1.0 + 1.0j), 0.0], "rpy_deg"),
    ],
)
def test_mapping_rejects_complex_rotation_parameterizations(
    field: str,
    value: list[object],
    message: str,
) -> None:
    payload = {"sensors": {"radar": {field: value}}}

    with warnings.catch_warnings():
        warnings.simplefilter("error", _COMPLEX_WARNING)
        with pytest.raises(ValueError, match=rf"{message} must contain real values"):
            calibration_from_mapping(payload)


def test_mapping_rejects_complex_time_offset_before_cast() -> None:
    payload = {
        "sensors": {
            "radar": {
                "time_offset_s": np.complex64(0.25 + 0.5j),
            }
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", _COMPLEX_WARNING)
        with pytest.raises(ValueError, match="time_offset_s.*must contain real values"):
            calibration_from_mapping(payload)


def test_apply_rejects_complex_points_before_cast() -> None:
    transform = RigidTransform.identity()
    points = np.array([[1.0 + 2.0j, 0.0, 0.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", _COMPLEX_WARNING)
        with pytest.raises(ValueError, match="xyz must contain real values"):
            transform.apply(points)


def test_rigid_transform_keeps_object_wrapped_real_values() -> None:
    transform = RigidTransform(
        rotation=np.array(np.eye(3), dtype=object),
        translation_m=np.array([1.0, 2.0, 3.0], dtype=object),
    )

    np.testing.assert_allclose(transform.rotation, np.eye(3))
    np.testing.assert_allclose(transform.translation_m, [1.0, 2.0, 3.0])


def test_mapping_keeps_opencv_style_real_matrix() -> None:
    payload = {
        "sensors": {
            "camera": {
                "rotation_matrix": {
                    "data": np.eye(3).reshape(-1).tolist(),
                    "rows": 3,
                    "cols": 3,
                },
                "translation_m": [1.0, 2.0, 3.0],
            }
        }
    }

    calibration = calibration_from_mapping(payload)
    transform = calibration.sensors["camera"].transform_sensor_to_world

    np.testing.assert_allclose(transform.rotation, np.eye(3))
    np.testing.assert_allclose(transform.translation_m, [1.0, 2.0, 3.0])
