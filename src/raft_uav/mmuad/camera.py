"""Camera detection bridges for MMUAD-style experiments.

This module does not run image detection.  It converts exported camera detector
CSV rows into 3D candidate points when per-detection depth or a fixed depth
proxy is available.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.calibration import RigidTransform
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraModel:
    source: str
    intrinsics: CameraIntrinsics
    transform_camera_to_world: RigidTransform
    time_offset_s: float = 0.0


def load_camera_models(path: Path) -> dict[str, CameraModel]:
    """Load simple camera intrinsics/extrinsics JSON or YAML.

    Supported layouts::

        {"cameras": {"cam0": {"fx": ..., "fy": ..., "cx": ..., "cy": ...}}}

    or a single-camera top-level object.  Rotation fields mirror the generic
    calibration file: ``rotation_matrix``, ``quaternion_wxyz`` or ``rpy_deg``.
    """

    payload = _load_json_or_yaml(path)
    cameras = payload.get("cameras", payload)
    if not isinstance(cameras, dict):
        raise ValueError("camera calibration must contain a mapping")
    models: dict[str, CameraModel] = {}
    if _looks_like_single_camera(cameras):
        cameras = {str(cameras.get("source", "camera")): cameras}
    for source, entry in cameras.items():
        if not isinstance(entry, dict):
            continue
        intrinsics = _intrinsics_from_camera_entry(entry)
        transform = _transform_from_camera_entry(entry)
        models[str(source).lower()] = CameraModel(
            source=str(source),
            intrinsics=intrinsics,
            transform_camera_to_world=transform,
            time_offset_s=float(entry.get("time_offset_s", 0.0)),
        )
    if not models:
        raise ValueError(f"no camera models found in {path}")
    return models


def load_camera_detections_csv_as_candidates(
    path: Path,
    *,
    camera_models: dict[str, CameraModel],
    source: str | None = None,
    default_source: str | None = None,
    sequence_id: str | None = None,
    fixed_depth_m: float | None = None,
    std_xy_m: float = 5.0,
    std_z_m: float = 10.0,
) -> CandidateFrame:
    """Convert exported camera detections into 3D candidate points.

    Detection rows may contain ``u_px``/``v_px`` or bounding-box aliases
    ``x1,y1,x2,y2``.  Depth must come from ``depth_m``/``range_m`` unless a
    ``fixed_depth_m`` fallback is supplied.  This is a detector-output bridge,
    not a camera detector.
    """

    frame = _normalize_camera_detection_columns(_read_delimited_table(path))
    if source is not None:
        frame["source"] = str(source)
    elif "source" not in frame.columns:
        frame["source"] = str(default_source or "camera")
    if sequence_id is not None:
        frame["sequence_id"] = str(sequence_id)
    elif "sequence_id" not in frame.columns:
        frame["sequence_id"] = Path(path).parent.name
    if "depth_m" not in frame.columns:
        if fixed_depth_m is None:
            raise ValueError("camera detections need depth_m/range_m or --camera-fixed-depth-m")
        frame["depth_m"] = float(fixed_depth_m)
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        time_s = float(row["time_s"])
        src = str(row["source"])
        model = _model_for_source(camera_models, src)
        if model is None:
            raise ValueError(f"no camera model found for source {src!r}")
        u = float(row["u_px"])
        v = float(row["v_px"])
        depth = float(row["depth_m"])
        xyz_camera = backproject_pixel_to_camera_xyz(u, v, depth, model.intrinsics)
        xyz_world = model.transform_camera_to_world.apply(xyz_camera)
        records.append(
            {
                "sequence_id": str(row["sequence_id"]),
                "time_s": time_s + model.time_offset_s,
                "source": src,
                "track_id": row.get("track_id", np.nan),
                "x_m": xyz_world[0],
                "y_m": xyz_world[1],
                "z_m": xyz_world[2],
                "std_xy_m": float(row.get("std_xy_m", std_xy_m)),
                "std_z_m": float(row.get("std_z_m", std_z_m)),
                "confidence": float(row.get("confidence", 1.0)),
                "class_name": str(row.get("class_name", "uav")),
            }
        )
    return CandidateFrame(normalize_candidate_columns(pd.DataFrame.from_records(records)))


def backproject_pixel_to_camera_xyz(
    u_px: float,
    v_px: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Back-project a pixel and metric depth into camera coordinates."""

    z = float(depth_m)
    x = (float(u_px) - intrinsics.cx) / intrinsics.fx * z
    y = (float(v_px) - intrinsics.cy) / intrinsics.fy * z
    return np.array([x, y, z], dtype=float)


def _normalize_camera_detection_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lower = {str(col).lower(): col for col in frame.columns}
    rename: dict[object, str] = {}
    aliases = {
        "time_s": ("time_s", "timestamp_s", "timestamp", "time", "t"),
        "sequence_id": ("sequence_id", "sequence", "seq", "scene_id"),
        "source": ("source", "camera", "camera_id", "sensor"),
        "track_id": ("track_id", "track", "id", "object_id"),
        "u_px": ("u_px", "u", "pixel_x", "center_u", "cx_px"),
        "v_px": ("v_px", "v", "pixel_y", "center_v", "cy_px"),
        "x1": ("x1", "xmin", "bbox_x1", "left"),
        "y1": ("y1", "ymin", "bbox_y1", "top"),
        "x2": ("x2", "xmax", "bbox_x2", "right"),
        "y2": ("y2", "ymax", "bbox_y2", "bottom"),
        "depth_m": ("depth_m", "range_m", "distance_m", "z_depth_m"),
        "confidence": ("confidence", "score", "probability"),
        "class_name": ("class_name", "uav_type", "class", "label", "category"),
        "std_xy_m": ("std_xy_m", "xy_std_m", "std_m"),
        "std_z_m": ("std_z_m", "z_std_m"),
    }
    for canonical, choices in aliases.items():
        if canonical in frame.columns:
            continue
        for alias in choices:
            original = lower.get(alias.lower())
            if original is not None:
                rename[original] = canonical
                break
    out = frame.rename(columns=rename).copy()
    if "u_px" not in out.columns or "v_px" not in out.columns:
        if {"x1", "y1", "x2", "y2"}.issubset(out.columns):
            out["u_px"] = (
                pd.to_numeric(out["x1"], errors="coerce")
                + pd.to_numeric(out["x2"], errors="coerce")
            ) / 2.0
            out["v_px"] = (
                pd.to_numeric(out["y1"], errors="coerce")
                + pd.to_numeric(out["y2"], errors="coerce")
            ) / 2.0
        else:
            raise ValueError("camera detection CSV needs u/v pixels or bbox x1/y1/x2/y2")
    if "time_s" not in out.columns:
        raise ValueError("camera detection CSV requires time_s/timestamp_s/time column")
    for col in ("time_s", "u_px", "v_px", "depth_m", "confidence", "std_xy_m", "std_z_m"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.loc[np.isfinite(out[["time_s", "u_px", "v_px"]]).all(axis=1)].copy()


def _model_for_source(models: dict[str, CameraModel], source: str) -> CameraModel | None:
    source_l = str(source).lower()
    if source_l in models:
        return models[source_l]
    if len(models) == 1:
        return next(iter(models.values()))
    for key, model in models.items():
        if source_l.startswith(key) or key.startswith(source_l):
            return model
    return None


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception:
            return json.loads(text)
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"camera calibration must be a mapping: {path}")
    return payload


def _read_delimited_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _looks_like_single_camera(payload: dict[str, Any]) -> bool:
    return (
        "fx" in payload
        or "intrinsics" in payload
        or any(
            key in payload
            for key in (
                "camera_matrix",
                "cameraMatrix",
                "intrinsic_matrix",
                "projection_matrix",
                "K",
                "P",
            )
        )
    )


def _intrinsics_from_camera_entry(entry: dict[str, Any]) -> CameraIntrinsics:
    intrinsics_payload = entry.get("intrinsics", entry)
    if all(key in intrinsics_payload for key in ("fx", "fy", "cx", "cy")):
        return CameraIntrinsics(
            fx=float(intrinsics_payload["fx"]),
            fy=float(intrinsics_payload["fy"]),
            cx=float(intrinsics_payload["cx"]),
            cy=float(intrinsics_payload["cy"]),
        )
    matrix = _intrinsic_matrix_from_entry(intrinsics_payload)
    if matrix is None and intrinsics_payload is not entry:
        matrix = _intrinsic_matrix_from_entry(entry)
    if matrix is None:
        raise ValueError("camera calibration entry needs fx/fy/cx/cy or a camera matrix")
    values = np.asarray(matrix, dtype=float)
    if values.shape == (3, 3):
        return CameraIntrinsics(
            fx=float(values[0, 0]),
            fy=float(values[1, 1]),
            cx=float(values[0, 2]),
            cy=float(values[1, 2]),
        )
    if values.shape == (3, 4):
        return CameraIntrinsics(
            fx=float(values[0, 0]),
            fy=float(values[1, 1]),
            cx=float(values[0, 2]),
            cy=float(values[1, 2]),
        )
    raise ValueError(f"camera intrinsics matrix must be 3x3 or 3x4, got {values.shape}")


def _intrinsic_matrix_from_entry(entry: dict[str, Any]) -> np.ndarray | None:
    from raft_uav.mmuad.calibration import _matrix_from_value

    for key in (
        "camera_matrix",
        "cameraMatrix",
        "intrinsic_matrix",
        "projection_matrix",
        "K",
        "P",
    ):
        if key in entry:
            return _matrix_from_value(entry[key])
    return None


def _transform_from_camera_entry(entry: dict[str, Any]) -> RigidTransform:
    from raft_uav.mmuad.calibration import _transform_from_entry

    return _transform_from_entry(entry)
