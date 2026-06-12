"""Camera detection bridges for MMUAD-style experiments.

This module does not run image detection.  It converts exported camera detector
table rows into 3D candidate points when per-detection depth or a fixed depth
proxy is available.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.calibration import RigidTransform
from raft_uav.mmuad.schema import (
    CandidateFrame,
    normalize_candidate_columns,
    normalize_time_column_aliases,
)


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


def load_camera_models_from_files(
    paths: Iterable[Path],
    *,
    source_hint_from_path: Callable[[Path], str | None] | None = None,
) -> dict[str, CameraModel]:
    """Load and merge camera models from one or more calibration/intrinsics files."""

    models: dict[str, CameraModel] = {}
    errors: list[str] = []
    for path in paths:
        try:
            loaded = load_camera_models(path)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if source_hint_from_path is not None:
            source_hint = source_hint_from_path(path)
            if source_hint is not None and len(loaded) == 1:
                model = next(iter(loaded.values()))
                loaded = {source_hint.lower(): replace(model, source=source_hint)}
        models.update(loaded)
    if not models:
        detail = "; ".join(errors)
        raise ValueError(f"no camera models found in camera calibration files: {detail}")
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

    Detection rows may contain ``u_px``/``v_px``, bounding-box aliases
    ``x1,y1,x2,y2``, COCO-style ``bbox=[x,y,width,height]``, or explicit
    ``bbox_xyxy=[x1,y1,x2,y2]``.  Depth must come from ``depth_m``/``range_m``
    unless a ``fixed_depth_m`` fallback is supplied. CSV/TSV/TXT and JSON
    row/table exports are supported. This is a detector-output bridge, not a
    camera detector.
    """

    return camera_detection_frame_to_candidates(
        _read_detection_table(path),
        camera_models=camera_models,
        source=source,
        default_source=default_source,
        sequence_id=sequence_id,
        default_sequence_id=Path(path).parent.name,
        fixed_depth_m=fixed_depth_m,
        std_xy_m=std_xy_m,
        std_z_m=std_z_m,
    )


def camera_detection_frame_to_candidates(
    frame: pd.DataFrame,
    *,
    camera_models: dict[str, CameraModel],
    source: str | None = None,
    default_source: str | None = None,
    sequence_id: str | None = None,
    default_sequence_id: str = "default",
    fixed_depth_m: float | None = None,
    std_xy_m: float = 5.0,
    std_z_m: float = 10.0,
) -> CandidateFrame:
    """Convert an exported camera-detection table frame into 3D candidates."""

    frame = _normalize_camera_detection_columns(frame)
    if source is not None:
        frame["source"] = str(source)
    elif "source" not in frame.columns:
        frame["source"] = str(default_source or "camera")
    if sequence_id is not None:
        frame["sequence_id"] = str(sequence_id)
    elif "sequence_id" not in frame.columns:
        frame["sequence_id"] = str(default_sequence_id)
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
    frame = normalize_time_column_aliases(frame, target="time_s")
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
    out = _expand_compact_bbox_columns(frame.rename(columns=rename).copy())
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
            raise ValueError("camera detection table needs u/v pixels or bbox geometry")
    if "time_s" not in out.columns:
        raise ValueError("camera detection table requires time_s/timestamp_s/time column")
    for col in ("time_s", "u_px", "v_px", "depth_m", "confidence", "std_xy_m", "std_z_m"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.loc[np.isfinite(out[["time_s", "u_px", "v_px"]]).all(axis=1)].copy()


_COMPACT_BBOX_XYWH_COLUMNS = ("bbox", "bbox_xywh", "xywh", "box_xywh")
_COMPACT_BBOX_XYXY_COLUMNS = ("bbox_xyxy", "xyxy", "box_xyxy")
_COMPACT_BBOX_COLUMNS = _COMPACT_BBOX_XYWH_COLUMNS + _COMPACT_BBOX_XYXY_COLUMNS


def _expand_compact_bbox_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if {"x1", "y1", "x2", "y2"}.issubset(frame.columns):
        return frame
    xyxy_col = _first_existing_column(frame, _COMPACT_BBOX_XYXY_COLUMNS)
    layout = "xyxy"
    compact_col = xyxy_col
    if compact_col is None:
        compact_col = _first_existing_column(frame, _COMPACT_BBOX_XYWH_COLUMNS)
        layout = "xywh"
    if compact_col is None:
        return frame
    out = frame.copy()
    boxes = out[compact_col].map(lambda value: _compact_bbox_to_xyxy(value, layout=layout))
    coords = pd.DataFrame(
        boxes.tolist(),
        columns=["x1", "y1", "x2", "y2"],
        index=out.index,
    )
    for col in coords.columns:
        if col not in out.columns:
            out[col] = coords[col]
    return out


def _first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> object | None:
    lower = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        original = lower.get(candidate.lower())
        if original is not None:
            return original
    return None


def _compact_bbox_to_xyxy(value: Any, *, layout: str) -> tuple[float, float, float, float]:
    mapped = _compact_bbox_mapping_to_xyxy(value)
    if mapped is not None:
        return mapped
    numbers = _numeric_sequence(value)
    if len(numbers) < 4:
        return (np.nan, np.nan, np.nan, np.nan)
    a, b, c, d = numbers[:4]
    if layout == "xyxy":
        return (a, b, c, d)
    return (a, b, a + c, b + d)


def _compact_bbox_mapping_to_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, dict):
        return None
    xyxy = _mapping_float_values(
        value,
        (
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
        ),
    )
    if xyxy is not None:
        return xyxy
    xywh = _mapping_float_values(
        value,
        (
            ("x", "y", "width", "height"),
            ("x", "y", "w", "h"),
            ("left", "top", "width", "height"),
            ("left", "top", "w", "h"),
        ),
    )
    if xywh is None:
        return None
    x, y, width, height = xywh
    return (x, y, x + width, y + height)


def _mapping_float_values(
    mapping: dict[Any, Any],
    key_groups: Iterable[tuple[str, str, str, str]],
) -> tuple[float, float, float, float] | None:
    lower = {str(key).lower(): value for key, value in mapping.items()}
    for key_group in key_groups:
        if not all(key in lower for key in key_group):
            continue
        values: list[float] = []
        try:
            values = [float(lower[key]) for key in key_group]
        except (TypeError, ValueError):
            continue
        return (values[0], values[1], values[2], values[3])
    return None


def _numeric_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return _numeric_sequence(json.loads(text))
        except json.JSONDecodeError:
            trimmed = text.strip("[]()")
            delimiter = "," if "," in trimmed else None
            parts = trimmed.split(delimiter)
            return _numeric_sequence(parts)
    if isinstance(value, dict):
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        numbers: list[float] = []
        for item in value:
            try:
                numbers.append(float(item))
            except (TypeError, ValueError):
                return []
        return numbers
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        return []
    return []


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


def _read_detection_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _read_json_detection_table(path)
    return _read_delimited_table(path)


def _read_json_detection_table(path: Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = _json_detection_records(payload)
    return _json_detection_records_to_frame(records, path=path)


def _json_detection_records(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in (
        "camera_detections",
        "image_detections",
        "detections",
        "annotations",
        "boxes",
        "bboxes",
        "rows",
        "data",
    ):
        nested = _mapping_get_case_insensitive(payload, key)
        if nested is not None:
            return _json_detection_records(nested)
    if _looks_like_detection_column_map(payload) or _looks_like_detection_row(payload):
        return payload
    return []


def _json_detection_records_to_frame(records: Any, *, path: Path | None = None) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, dict):
        if _looks_like_detection_column_map(records):
            return pd.DataFrame(records)
        if _looks_like_detection_row(records):
            return pd.DataFrame.from_records([records])
    if isinstance(records, list):
        if not records:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in records):
            return pd.DataFrame.from_records(records)
    label = str(path) if path is not None else "JSON payload"
    raise ValueError(f"camera detection JSON table {label} does not contain row objects")


def _mapping_get_case_insensitive(mapping: dict[Any, Any], key: str) -> Any | None:
    for candidate, value in mapping.items():
        if str(candidate).lower() == key.lower():
            return value
    return None


_DETECTION_HINT_KEYS = {
    "time_s",
    "timestamp",
    "timestamp_s",
    "timestamp_ns",
    "timestamp_ms",
    "sec",
    "nanosec",
    "u_px",
    "u",
    "pixel_x",
    "center_u",
    "v_px",
    "v",
    "pixel_y",
    "center_v",
    "x1",
    "xmin",
    "bbox_x1",
    "y1",
    "ymin",
    "bbox_y1",
    "x2",
    "xmax",
    "bbox_x2",
    "y2",
    "ymax",
    "bbox_y2",
    "bbox",
    "bbox_xywh",
    "xywh",
    "box_xywh",
    "bbox_xyxy",
    "xyxy",
    "box_xyxy",
    "depth_m",
    "range_m",
}


def _looks_like_detection_row(payload: dict[Any, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    return bool(keys.intersection(_DETECTION_HINT_KEYS))


def _looks_like_detection_column_map(payload: dict[Any, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    if not keys.intersection(_DETECTION_HINT_KEYS):
        return False
    return any(
        isinstance(value, (list, tuple))
        for key, value in payload.items()
        if str(key).lower() in _DETECTION_HINT_KEYS
        and str(key).lower() not in _COMPACT_BBOX_COLUMNS
    )


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
                "k",
                "P",
                "p",
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
        "k",
        "P",
        "p",
    ):
        if key in entry:
            return _matrix_from_value(entry[key])
    return None


def _transform_from_camera_entry(entry: dict[str, Any]) -> RigidTransform:
    from raft_uav.mmuad.calibration import _transform_from_entry

    return _transform_from_entry(entry)
