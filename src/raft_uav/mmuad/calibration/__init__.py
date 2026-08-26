"""Compatibility validation for MMUAD calibration transforms.

The maintained implementation lives in the sibling ``calibration.py`` module.
This package preserves the public import path while rejecting complex-valued
calibration inputs before NumPy can silently discard their imaginary components,
rejecting ambiguous or blank sensor names before calibration selection becomes
order dependent, and rejecting malformed 4x4 projective matrices before their
non-homogeneous final row can be silently ignored.
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
_HOMOGENEOUS_LAST_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
_HOMOGENEOUS_LAST_ROW_ATOL = 1.0e-9


def _contains_complex(value: Any) -> bool:
    """Return whether a nested array-like value contains a complex scalar."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, dict):
        return any(item is not value and _contains_complex(item) for item in value.values())
    try:
        values = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.iscomplexobj(values):
        return True
    if values.dtype != object:
        return False
    return any(item is not value and _contains_complex(item) for item in values.flat)


def _reject_complex(value: Any, *, name: str) -> None:
    if _contains_complex(value):
        raise ValueError(f"{name} must contain real values")


def _normalized_sensor_name(value: object) -> str:
    """Return the logical calibration sensor name used during lookup."""

    return str(value).strip().lower()


def _validate_sensor_names(payload: dict[str, Any]) -> None:
    """Reject sensor names whose normalized lookup identity is ambiguous."""

    sensors_payload = payload.get("sensors", payload)
    if not isinstance(sensors_payload, dict):
        return
    sensors_are_nested = sensors_payload is not payload
    normalized: dict[str, object] = {}
    for source in sensors_payload:
        sensor_name = _normalized_sensor_name(source)
        if (
            not sensors_are_nested
            and sensor_name in _IMPL._CALIBRATION_METADATA_KEYS
        ):
            continue
        if not sensor_name:
            raise ValueError("calibration sensor names must not be blank")
        if sensor_name in normalized:
            raise ValueError(
                "calibration sensor names are ambiguous after trimming whitespace "
                f"and ignoring case: {normalized[sensor_name]!r} and {source!r}"
            )
        normalized[sensor_name] = source


def _validate_homogeneous_transform_matrix(matrix: np.ndarray) -> np.ndarray:
    """Require canonical affine homogeneous coordinates for every 4x4 transform."""

    values = np.asarray(matrix, dtype=float)
    if values.shape != (4, 4):
        return values
    last_row = values[3]
    if not np.isfinite(last_row).all() or not np.allclose(
        last_row,
        _HOMOGENEOUS_LAST_ROW,
        rtol=0.0,
        atol=_HOMOGENEOUS_LAST_ROW_ATOL,
    ):
        raise ValueError(
            "4x4 transform matrix must end with homogeneous row [0, 0, 0, 1]"
        )
    return values


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
    values = _validate_homogeneous_transform_matrix(matrix)
    return _ORIGINAL_TRANSFORM_FROM_MATRIX(values)


def _load_single_matrix_calibration(path: Path) -> Any:
    """Load one rigid 4x4 transform without discarding a projective final row."""

    path = Path(path)
    values = np.loadtxt(
        path,
        delimiter="," if path.suffix.lower() == ".csv" else None,
    )
    values = np.asarray(values, dtype=float)
    if values.shape == (4, 4):
        matrix = values
    elif values.size == 16:
        matrix = values.reshape(4, 4)
    else:
        raise ValueError("text calibration must contain one 4x4 matrix")
    matrix = _validate_homogeneous_transform_matrix(matrix)
    return _IMPL.CalibrationSet(
        sensors={
            "default": SensorCalibration(
                source="default",
                transform_sensor_to_world=RigidTransform(
                    rotation=matrix[:3, :3],
                    translation_m=matrix[:3, 3],
                ),
            )
        },
        world_frame="world",
    )


def calibration_from_mapping(payload: dict[str, Any]):
    """Build calibrations after validating names and cast-before-dispatch values."""

    _validate_sensor_names(payload)
    sensors_payload = payload.get("sensors", payload)
    if isinstance(sensors_payload, dict):
        for source, entry in sensors_payload.items():
            if not isinstance(entry, dict):
                continue
            if "time_offset_s" in entry:
                _reject_complex(entry["time_offset_s"], name=f"time_offset_s for {source!r}")
            if "quaternion_wxyz" in entry:
                _reject_complex(entry["quaternion_wxyz"], name="quaternion")
            if "rpy_deg" in entry:
                _reject_complex(entry["rpy_deg"], name="rpy_deg")
    return _ORIGINAL_CALIBRATION_FROM_MAPPING(payload)


_IMPL.RigidTransform = RigidTransform
_IMPL.SensorCalibration = SensorCalibration
_IMPL.calibration_from_mapping = calibration_from_mapping
_IMPL._matrix_from_value = _matrix_from_value
_IMPL._reshape_flat_matrix = _reshape_flat_matrix
_IMPL._rotation_from_quaternion_wxyz = _rotation_from_quaternion_wxyz
_IMPL._rotation_from_rpy_deg = _rotation_from_rpy_deg
_IMPL._transform_from_matrix = _transform_from_matrix
_IMPL._load_single_matrix_calibration = _load_single_matrix_calibration

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_contains_complex"] = _contains_complex
globals()["_reject_complex"] = _reject_complex
globals()["_normalized_sensor_name"] = _normalized_sensor_name
globals()["_validate_sensor_names"] = _validate_sensor_names
globals()["_validate_homogeneous_transform_matrix"] = (
    _validate_homogeneous_transform_matrix
)
globals()["RigidTransform"] = RigidTransform
globals()["SensorCalibration"] = SensorCalibration
globals()["calibration_from_mapping"] = calibration_from_mapping
globals()["_matrix_from_value"] = _matrix_from_value
globals()["_reshape_flat_matrix"] = _reshape_flat_matrix
globals()["_rotation_from_quaternion_wxyz"] = _rotation_from_quaternion_wxyz
globals()["_rotation_from_rpy_deg"] = _rotation_from_rpy_deg
globals()["_transform_from_matrix"] = _transform_from_matrix
globals()["_load_single_matrix_calibration"] = _load_single_matrix_calibration

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
