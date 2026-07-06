"""Regression tests for finite MMUAD calibration validation."""

from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.calibration import RigidTransform, calibration_from_mapping


def test_rigid_transform_rejects_nonfinite_rotation() -> None:
    rotation = np.eye(3)
    rotation[0, 0] = np.nan

    with pytest.raises(ValueError, match="rotation must contain finite values"):
        RigidTransform(rotation=rotation, translation_m=np.zeros(3))


def test_rigid_transform_rejects_nonfinite_translation() -> None:
    with pytest.raises(ValueError, match="translation_m must contain finite values"):
        RigidTransform(rotation=np.eye(3), translation_m=[0.0, np.inf, 0.0])


def test_calibration_rejects_nonfinite_quaternion() -> None:
    payload = {"sensors": {"camera": {"quaternion_wxyz": [np.nan, 0.0, 0.0, 1.0]}}}

    with pytest.raises(ValueError, match="quaternion must contain finite values"):
        calibration_from_mapping(payload)


def test_calibration_rejects_nonfinite_rpy() -> None:
    payload = {"sensors": {"radar": {"rpy_deg": [0.0, np.inf, 0.0]}}}

    with pytest.raises(ValueError, match="rpy_deg must contain finite values"):
        calibration_from_mapping(payload)


def test_calibration_rejects_nonfinite_time_offset() -> None:
    payload = {"sensors": {"radar": {"time_offset_s": np.nan}}}

    with pytest.raises(ValueError, match="time_offset_s must be finite"):
        calibration_from_mapping(payload)
