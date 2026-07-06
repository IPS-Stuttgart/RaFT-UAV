"""Calibration and extrinsic-transform helpers for MMUAD-style exports.

The official challenge archive may expose calibration in several concrete file
formats.  This module intentionally uses a small JSON interchange format that
can be generated from an official parser once the raw layout is available.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import (
    CandidateFrame,
    TruthFrame,
    normalize_candidate_columns,
    normalize_truth_columns,
)

_CALIBRATION_METADATA_KEYS = {"metadata", "schema", "version", "world_frame"}


@dataclass(frozen=True)
class RigidTransform:
    """A right-handed rigid transform from a sensor frame into the world frame."""

    rotation: np.ndarray
    translation_m: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation_m, dtype=float).reshape(3)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
        if not np.isfinite(rotation).all():
            raise ValueError("rotation must contain finite values")
        if not np.isfinite(translation).all():
            raise ValueError("translation_m must contain finite values")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(rotation=np.eye(3), translation_m=np.zeros(3))

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz, dtype=float)
        if points.ndim == 1:
            return self.rotation @ points.reshape(3) + self.translation_m
        return points @ self.rotation.T + self.translation_m

    def inverse(self) -> "RigidTransform":
        rotation_inv = self.rotation.T
        translation_inv = -(rotation_inv @ self.translation_m)
        return RigidTransform(rotation=rotation_inv, translation_m=translation_inv)


@dataclass(frozen=True)
class SensorCalibration:
    """Calibration entry for one sensor/modality."""

    source: str
    transform_sensor_to_world: RigidTransform
    time_offset_s: float = 0.0

    def __post_init__(self) -> None:
        time_offset_s = float(self.time_offset_s)
        if not np.isfinite(time_offset_s):
            raise ValueError("time_offset_s must be finite")
        object.__setattr__(self, "time_offset_s", time_offset_s)


@dataclass(frozen=True)
class CalibrationSet:
    """Collection of source-specific calibrations."""

    sensors: dict[str, SensorCalibration]
    world_frame: str = "world"

    def get(self, source: str) -> SensorCalibration | None:
        """Return the best calibration entry for a candidate source name.

        Candidate sources can be more specific than calibration keys, e.g.
        ``radar_enhance_pcl_clusters`` should match ``radar_enhance_pcl``.
        When both a generic key (``radar``) and a specific key match, prefer the
        longest key; otherwise JSON insertion order could make the generic entry
        shadow the specific calibration.
        """

        source_l = str(source).lower()
        for key, value in self.sensors.items():
            if source_l == str(key).lower():
                return value
        matches: list[tuple[int, SensorCalibration]] = []
        for key, value in self.sensors.items():
            key_l = str(key).lower()
            if source_l.startswith(key_l) or key_l.startswith(source_l):
                matches.append((len(key_l), value))
        if not matches:
            return None
        return max(matches, key=lambda item: item[0])[1]


def load_calibration_auto(path: Path) -> CalibrationSet:
    """Load JSON/YAML/TXT calibration interchange files when possible.

    YAML support uses PyYAML when installed.  Plain text support expects a single
    4x4 matrix and assigns it to a default sensor.  This helper is intentionally
    conservative; unknown official formats should first be inspected rather than
    silently misread.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_calibration_json(path)
    if suffix in {".yaml", ".yml"}:
        payload = _load_yaml_or_json(path)
        return calibration_from_mapping(payload)
    if suffix in {".txt", ".csv"}:
        return _load_single_matrix_calibration(path)
    raise ValueError(f"unsupported calibration extension: {path.suffix}")


def load_calibration_file(path: Path) -> CalibrationSet:
    """Backward-compatible alias for calibration interchange file loading."""

    return load_calibration_auto(path)


def load_calibration_json(path: Path) -> CalibrationSet:
    """Load a simple MMUAD calibration interchange JSON.

    Supported per-sensor transform fields:

    - ``rotation_matrix``: 3x3 matrix;
    - ``quaternion_wxyz``: quaternion in ``[w, x, y, z]`` order;
    - ``rpy_deg``: roll/pitch/yaw degrees with yaw about +z;
    - ``translation_m``: length-3 translation.

    Example::

        {
          "world_frame": "leica_world",
          "sensors": {
            "radar": {
              "translation_m": [1, 2, 3],
              "rpy_deg": [0, 0, 90],
              "time_offset_s": -0.01
            }
          }
        }
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return calibration_from_mapping(payload)


def calibration_from_mapping(payload: dict[str, Any]) -> CalibrationSet:
    """Build a calibration set from a mapping payload."""

    sensors_payload = payload.get("sensors", payload)
    if not isinstance(sensors_payload, dict):
        raise ValueError("calibration JSON must contain a sensors mapping")
    sensors_are_nested = sensors_payload is not payload
    sensors: dict[str, SensorCalibration] = {}
    for source, entry in sensors_payload.items():
        if not sensors_are_nested and str(source).lower() in _CALIBRATION_METADATA_KEYS:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"calibration entry for {source!r} must be an object")
        transform = _transform_from_entry(entry)
        sensors[str(source).lower()] = SensorCalibration(
            source=str(source),
            transform_sensor_to_world=transform,
            time_offset_s=float(entry.get("time_offset_s", 0.0)),
        )
    if not sensors:
        raise ValueError("calibration JSON must contain at least one sensor calibration entry")
    return CalibrationSet(sensors=sensors, world_frame=str(payload.get("world_frame", "world")))


def transform_candidate_frame(
    frame: CandidateFrame,
    calibration: CalibrationSet,
    *,
    apply_time_offsets: bool = True,
    output_source_suffix: str | None = None,
) -> CandidateFrame:
    """Transform candidate coordinates into the calibration world frame."""

    rows = frame.rows.copy()
    if rows.empty:
        return frame
    transformed_parts: list[pd.DataFrame] = []
    for source, group in rows.groupby("source", sort=False):
        group = group.copy()
        sensor = calibration.get(str(source))
        if sensor is not None:
            xyz = group[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
            out_xyz = sensor.transform_sensor_to_world.apply(xyz)
            group["x_m"] = out_xyz[:, 0]
            group["y_m"] = out_xyz[:, 1]
            group["z_m"] = out_xyz[:, 2]
            if apply_time_offsets:
                group["time_s"] = group["time_s"].astype(float) + sensor.time_offset_s
            group["calibration_applied"] = True
            group["calibration_world_frame"] = calibration.world_frame
            if output_source_suffix:
                group["source"] = group["source"].astype(str) + output_source_suffix
        else:
            group["calibration_applied"] = False
            group["calibration_world_frame"] = calibration.world_frame
        transformed_parts.append(group)
    rows = pd.concat(transformed_parts, ignore_index=True)
    return CandidateFrame(normalize_candidate_columns(rows))


def transform_truth_frame(
    frame: TruthFrame,
    transform: RigidTransform,
) -> TruthFrame:
    """Transform truth positions with a global rigid transform."""

    rows = frame.rows.copy()
    if rows.empty:
        return frame
    xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    out_xyz = transform.apply(xyz)
    rows["x_m"] = out_xyz[:, 0]
    rows["y_m"] = out_xyz[:, 1]
    rows["z_m"] = out_xyz[:, 2]
    return TruthFrame(normalize_truth_columns(rows))


def _calibration_from_payload(payload: dict[str, Any]) -> CalibrationSet:
    return calibration_from_mapping(payload)


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return json.loads(text)
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"calibration YAML must contain a mapping: {path}")
    return payload


def _transform_from_entry(entry: dict[str, Any]) -> RigidTransform:
    transform = _transform_matrix_from_entry(entry)
    if transform is not None:
        return _transform_from_matrix(transform)
    translation = _translation_from_entry(entry)
    if "rotation_matrix" in entry:
        rotation = _matrix_from_value(entry["rotation_matrix"])
    elif "rotation" in entry:
        rotation = _matrix_from_value(entry["rotation"])
    elif "R" in entry:
        rotation = _matrix_from_value(entry["R"])
    elif "quaternion_wxyz" in entry:
        rotation = _rotation_from_quaternion_wxyz(np.asarray(entry["quaternion_wxyz"], dtype=float))
    elif "rpy_deg" in entry:
        rotation = _rotation_from_rpy_deg(np.asarray(entry["rpy_deg"], dtype=float))
    else:
        rotation = np.eye(3)
    return RigidTransform(rotation=rotation, translation_m=translation)


def _rotation_from_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    if not np.isfinite(q).all():
        raise ValueError("quaternion must contain finite values")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("quaternion must be nonzero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotation_from_rpy_deg(rpy_deg: np.ndarray) -> np.ndarray:
    rpy = np.asarray(rpy_deg, dtype=float).reshape(3)
    if not np.isfinite(rpy).all():
        raise ValueError("rpy_deg must contain finite values")
    roll, pitch, yaw = np.deg2rad(rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _load_single_matrix_calibration(path: Path) -> CalibrationSet:
    values = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    values = np.asarray(values, dtype=float)
    if values.shape == (4, 4):
        rotation = values[:3, :3]
        translation = values[:3, 3]
    elif values.size == 16:
        matrix = values.reshape(4, 4)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
    else:
        raise ValueError("text calibration must contain one 4x4 matrix")
    return CalibrationSet(
        sensors={
            "default": SensorCalibration(
                source="default",
                transform_sensor_to_world=RigidTransform(rotation=rotation, translation_m=translation),
            )
        },
        world_frame="world",
    )


def _transform_matrix_from_entry(entry: dict[str, Any]) -> np.ndarray | None:
    for key in (
        "transform_matrix",
        "extrinsic_matrix",
        "T_sensor_to_world",
        "T_camera_to_world",
        "T_cam_world",
        "T",
        "matrix",
        "transform",
    ):
        if key not in entry:
            continue
        matrix = _matrix_from_value(entry[key])
        if matrix.shape in {(3, 4), (4, 4)}:
            return matrix
    return None


def _transform_from_matrix(matrix: np.ndarray) -> RigidTransform:
    values = np.asarray(matrix, dtype=float)
    if values.shape == (4, 4):
        return RigidTransform(rotation=values[:3, :3], translation_m=values[:3, 3])
    if values.shape == (3, 4):
        return RigidTransform(rotation=values[:, :3], translation_m=values[:, 3])
    raise ValueError(f"transform matrix must be 3x4 or 4x4, got {values.shape}")


def _translation_from_entry(entry: dict[str, Any]) -> np.ndarray:
    for key in ("translation_m", "translation", "translation_vector", "tvec", "t", "T"):
        if key not in entry:
            continue
        value = _matrix_from_value(entry[key])
        flat = np.asarray(value, dtype=float).reshape(-1)
        if flat.size == 3:
            return flat
    return np.zeros(3)


def _matrix_from_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        data = value.get("data", value.get("values", value.get("value")))
        if data is None:
            raise ValueError("matrix mapping must contain data/values")
        arr = np.asarray(data, dtype=float)
        rows = int(value.get("rows", 0) or 0)
        cols = int(value.get("cols", value.get("columns", 0)) or 0)
        if rows > 0 and cols > 0:
            return arr.reshape(rows, cols)
        return _reshape_flat_matrix(arr)
    arr = np.asarray(value, dtype=float)
    return _reshape_flat_matrix(arr)


def _reshape_flat_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim >= 2:
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    if arr.size == 12:
        return arr.reshape(3, 4)
    if arr.size == 9:
        return arr.reshape(3, 3)
    return arr
