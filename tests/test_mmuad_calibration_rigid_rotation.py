"""Regression tests for rigid MMUAD calibration rotations."""

from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.calibration import RigidTransform, calibration_from_mapping


def test_rigid_transform_rejects_scaled_rotation() -> None:
    rotation = np.diag([2.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="rotation must be orthonormal"):
        RigidTransform(rotation=rotation, translation_m=np.zeros(3))


def test_calibration_rejects_left_handed_rotation() -> None:
    payload = {
        "sensors": {
            "camera": {
                "rotation_matrix": np.diag([-1.0, 1.0, 1.0]).tolist(),
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=r"rotation must be right-handed with determinant \+1",
    ):
        calibration_from_mapping(payload)


def test_rigid_transform_inverse_round_trip() -> None:
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = RigidTransform(
        rotation=rotation,
        translation_m=np.array([4.0, -2.0, 7.0]),
    )
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5]])

    restored = transform.inverse().apply(transform.apply(points))

    np.testing.assert_allclose(restored, points, rtol=0.0, atol=1.0e-12)
