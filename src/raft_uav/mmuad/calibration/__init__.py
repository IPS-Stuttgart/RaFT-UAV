"""Compatibility validation for MMUAD calibration transforms.

The maintained implementation lives in the sibling ``calibration.py`` module.
This package preserves the public import path while rejecting complex-valued
calibration inputs before NumPy can silently discard their imaginary components
during real-valued coercion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "calibration.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._calibration_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load calibration implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RIGID_TRANSFORM = _IMPL.RigidTransform
_ORIGINAL_SENSOR_CALIBRATION = _IMPL.SensorCalibration
_ORIGINAL_CALIBRATION_FROM_MAPPING = _IMPL.calibration_from_mapping
_ORIGINAL_MATRIX_FROM_VALUE = _IMPL._matrix_from_value
_ORIGINAL_RESHAPE_FLAT_MATRIX = _IMPL._reshape_flat_matrix
_ORIGINAL_ROTATION_FROM_QUATERNION = _IMPL._rotation_from_quaternion_wxyz
_ORIGINAL_ROTATION_FROM_RPY = _IMPL._rotation_from_rpy_deg
_ORIGINAL_TRANSFORM_FROM_MATRIX = _IMPL._transform_from_matrix


def _contains_complex(value: Any) -> bool:
    """Return whether an array-like value contains a complex scalar."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    try:
        values = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.iscomplexobj(values):
        return True
    if values.dtype != object:
        return False
    return any(_contains_complex(item) for item in values.flat)


def _reject_complex(value: Any, *, name: str) -> None:
    if _contains_complex(value):
        raise ValueError(f"{name} must contain real values")


class RigidTransform(_ORIGINAL_RIGID_TRANSFORM):
    """Rigid transform that rejects lossy complex-to-real coercion."""

    def __post_init__(self) -> None:
        _reject_complex(self.rotation, name="rotation")
        _reject_complex(self.translation_m, name="translation_m")
        super().__post_init__()

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        _reject_complex(xyz, name="xyz")
        return super().apply(xyz)


class SensorCalibration(_ORIGINAL_SENSOR_CALIBRATION):
    """Sensor calibration that requires a real-valued clock offset."""

    def __post_init__(self) -> None:
        _reject_complex(self.time_offset_s, name="time_offset_s")
        super().__post_init__()


def _matrix_from_value(value: Any) -> np.ndarray:
    _reject_complex(value, name="calibration matrix")
    return _ORIGINAL_MATRIX_FROM_VALUE(value)


def _reshape_flat_matrix(values: np.ndarray) -> np.ndarray:
    _reject_complex(values, name="calibration matrix")
    return _ORIGINAL_RESHAPE_FLAT_MATRIX(values)


def _rotation_from_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    _reject_complex(q, name="quaternion")
    return _ORIGINAL_ROTATION_FROM_QUATERNION(q)


def _rotation_from_rpy_deg(rpy_deg: np.ndarray) -> np.ndarray:
    _reject_complex(rpy_deg, name="rpy_deg")
    return _ORIGINAL_ROTATION_FROM_RPY(rpy_deg)


def _transform_from_matrix(matrix: np.ndarray) -> RigidTransform:
    _reject_complex(matrix, name="transform matrix")
    return _ORIGINAL_TRANSFORM_FROM_MATRIX(matrix)


def calibration_from_mapping(payload: dict[str, Any]):
    """Build calibrations after validating real-valued clock offsets."""

    sensors_payload = payload.get("sensors", payload)
    if isinstance(sensors_payload, dict):
        for source, entry in sensors_payload.items():
            if isinstance(entry, dict) and "time_offset_s" in entry:
                _reject_complex(entry["time_offset_s"], name=f"time_offset_s for {source!r}")
    return _ORIGINAL_CALIBRATION_FROM_MAPPING(payload)


_IMPL.RigidTransform = RigidTransform
_IMPL.SensorCalibration = SensorCalibration
_IMPL.calibration_from_mapping = calibration_from_mapping
_IMPL._matrix_from_value = _matrix_from_value
_IMPL._reshape_flat_matrix = _reshape_flat_matrix
_IMPL._rotation_from_quaternion_wxyz = _rotation_from_quaternion_wxyz
_IMPL._rotation_from_rpy_deg = _rotation_from_rpy_deg
_IMPL._transform_from_matrix = _transform_from_matrix

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_contains_complex"] = _contains_complex
globals()["_reject_complex"] = _reject_complex
globals()["RigidTransform"] = RigidTransform
globals()["SensorCalibration"] = SensorCalibration
globals()["calibration_from_mapping"] = calibration_from_mapping
globals()["_matrix_from_value"] = _matrix_from_value
globals()["_reshape_flat_matrix"] = _reshape_flat_matrix
globals()["_rotation_from_quaternion_wxyz"] = _rotation_from_quaternion_wxyz
globals()["_rotation_from_rpy_deg"] = _rotation_from_rpy_deg
globals()["_transform_from_matrix"] = _transform_from_matrix

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
