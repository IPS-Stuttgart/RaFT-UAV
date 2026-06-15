"""Camera detection bridges for MMUAD-style experiments.

This module does not run image detection.  It converts exported camera detector
table rows or detector sidecar labels into 3D candidate points when
per-detection depth or a fixed depth proxy is available.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import struct
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.calibration import RigidTransform
from raft_uav.mmuad.io import JSON_TABLE_SUFFIXES, data_file_suffix, read_json_export_payload
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
    row/table exports are supported, including COCO-style JSON objects with
    top-level ``images`` and ``annotations`` arrays. Detection2D-style JSON rows
    can also carry metric depth on nested ``bbox.center.position.z`` /
    ``bbox.center.z`` fields.
    YOLO-style ``class cx cy width height [confidence]`` TXT sidecars are also
    accepted when a same-stem image is present for normalized pixel scaling.
    This is a detector-output bridge, not a camera detector.
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
    else:
        frame["source"] = _fill_missing_text(
            frame["source"],
            default_text=str(default_source or "camera"),
        )
    if sequence_id is not None:
        frame["sequence_id"] = str(sequence_id)
    elif "sequence_id" not in frame.columns:
        frame["sequence_id"] = str(default_sequence_id)
    else:
        frame["sequence_id"] = _fill_missing_text(
            frame["sequence_id"],
            default_text=str(default_sequence_id),
        )
    if "depth_m" not in frame.columns:
        if fixed_depth_m is None:
            raise ValueError("camera detections need depth_m/range_m or --camera-fixed-depth-m")
        frame["depth_m"] = float(fixed_depth_m)
    elif fixed_depth_m is not None:
        frame["depth_m"] = pd.to_numeric(frame["depth_m"], errors="coerce").fillna(
            float(fixed_depth_m)
        )
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
    return CandidateFrame(
        normalize_candidate_columns(
            pd.DataFrame.from_records(records),
            default_sequence_id=default_sequence_id,
            default_source=str(source or default_source or "camera"),
        )
    )


def _fill_missing_text(values: pd.Series, *, default_text: str) -> pd.Series:
    text = values.where(values.notna(), default_text).astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, str(default_text))


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
        "u_px": (
            "u_px",
            "u",
            "pixel_x",
            "center_u",
            "center_x",
            "center_x_px",
            "bbox_center_x",
            "bbox_cx",
            "cx_px",
        ),
        "v_px": (
            "v_px",
            "v",
            "pixel_y",
            "center_v",
            "center_y",
            "center_y_px",
            "bbox_center_y",
            "bbox_cy",
            "cy_px",
        ),
        "x1": ("x1", "xmin", "bbox_x1", "left"),
        "y1": ("y1", "ymin", "bbox_y1", "top"),
        "x2": ("x2", "xmax", "bbox_x2", "right"),
        "y2": ("y2", "ymax", "bbox_y2", "bottom"),
        "depth_m": ("depth_m", "depth", "range_m", "distance_m", "z_depth_m"),
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
    if data_file_suffix(path) == ".tsv":
        return pd.read_csv(path, sep="\t")
    if data_file_suffix(path) == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _read_detection_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if data_file_suffix(path) == ".txt" and _looks_like_yolo_label_file(path):
        return _read_yolo_label_table(path)
    if data_file_suffix(path) in JSON_TABLE_SUFFIXES:
        return _read_json_detection_table(path)
    return _read_delimited_table(path)


YOLO_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _looks_like_yolo_label_file(path: Path) -> bool:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return False
    observed = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) not in {5, 6}:
            return False
        try:
            [float(part) for part in parts]
        except ValueError:
            return False
        observed = True
    return observed


def _read_yolo_label_table(path: Path) -> pd.DataFrame:
    image_path = _same_stem_image_path(path)
    image_size = _image_size_px(image_path) if image_path is not None else None
    rows: list[dict[str, Any]] = []
    time_s = _timestamp_from_stem(path)
    for line_idx, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [float(part) for part in stripped.split()]
        if len(parts) not in {5, 6}:
            continue
        class_id, center_x, center_y, width, height = parts[:5]
        confidence = parts[5] if len(parts) == 6 else 1.0
        if _looks_normalized_box(center_x, center_y, width, height):
            if image_size is None:
                raise ValueError(
                    f"YOLO label file {path} uses normalized boxes but no same-stem "
                    "image with readable dimensions was found"
                )
            image_width, image_height = image_size
            center_x *= image_width
            width *= image_width
            center_y *= image_height
            height *= image_height
        x1 = center_x - (width / 2.0)
        y1 = center_y - (height / 2.0)
        x2 = center_x + (width / 2.0)
        y2 = center_y + (height / 2.0)
        rows.append(
            {
                "time_s": time_s,
                "u_px": center_x,
                "v_px": center_y,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "class_name": str(int(class_id)) if float(class_id).is_integer() else str(class_id),
                "track_id": f"{Path(path).stem}:{line_idx}",
                "image_file": str(image_path) if image_path is not None else "",
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "time_s",
            "u_px",
            "v_px",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "class_name",
            "track_id",
            "image_file",
        ],
    )


def _same_stem_image_path(path: Path) -> Path | None:
    stem = Path(path).stem
    directory = Path(path).parent
    for suffix in YOLO_IMAGE_SUFFIXES:
        for candidate in (directory / f"{stem}{suffix}", directory / f"{stem}{suffix.upper()}"):
            if candidate.exists():
                return candidate
    return None


def _looks_normalized_box(center_x: float, center_y: float, width: float, height: float) -> bool:
    values = (center_x, center_y, width, height)
    return all(np.isfinite(value) for value in values) and all(0.0 <= value <= 1.0 for value in values)


def _timestamp_from_stem(path: Path) -> float:
    timestamp = _timestamp_from_stem_or_none(path)
    return 0.0 if timestamp is None else timestamp


def _timestamp_from_stem_or_none(path: Path) -> float | None:
    import re

    tokens = re.findall(r"[-+]?\d*\.?\d+", Path(path).stem)
    if not tokens:
        return None
    return float(tokens[-1])


def _image_size_px(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        return _png_size_px(raw)
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_size_px(raw)
    if suffix == ".bmp":
        return _bmp_size_px(raw)
    return None


def _png_size_px(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return (int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big"))


def _jpeg_size_px(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 4 or raw[:2] != b"\xff\xd8":
        return None
    offset = 2
    while offset + 9 <= len(raw):
        if raw[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(raw) and raw[offset] == 0xFF:
            offset += 1
        if offset >= len(raw):
            return None
        marker = raw[offset]
        offset += 1
        if marker in {0xD8, 0xD9}:
            continue
        if offset + 2 > len(raw):
            return None
        segment_length = int.from_bytes(raw[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(raw):
            return None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            height = int.from_bytes(raw[offset + 3 : offset + 5], "big")
            width = int.from_bytes(raw[offset + 5 : offset + 7], "big")
            return (width, height)
        offset += segment_length
    return None


def _bmp_size_px(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 26 or raw[:2] != b"BM":
        return None
    width = struct.unpack_from("<i", raw, 18)[0]
    height = abs(struct.unpack_from("<i", raw, 22)[0])
    if width <= 0 or height <= 0:
        return None
    return (int(width), int(height))


def _read_json_detection_table(path: Path) -> pd.DataFrame:
    payload = read_json_export_payload(path)
    coco_frame = _coco_detection_frame(payload)
    if coco_frame is not None:
        return coco_frame
    records = _json_detection_records(payload)
    return _json_detection_records_to_frame(records, path=path)


def _coco_detection_frame(payload: Any) -> pd.DataFrame | None:
    if not isinstance(payload, dict):
        return None
    images = _mapping_get_case_insensitive(payload, "images")
    annotations = _mapping_get_case_insensitive(payload, "annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        return None
    image_rows = [item for item in images if isinstance(item, dict)]
    annotation_rows = [item for item in annotations if isinstance(item, dict)]
    if not image_rows or not annotation_rows:
        return pd.DataFrame()
    image_by_id = {
        _coco_key(_first_mapping_value(image, ("id", "image_id", "file_name", "file"))): image
        for image in image_rows
    }
    categories = _mapping_get_case_insensitive(payload, "categories")
    category_by_id = _coco_category_map(categories)
    rows: list[dict[str, Any]] = []
    for annotation_idx, annotation in enumerate(annotation_rows):
        image_key = _coco_key(
            _first_mapping_value(annotation, ("image_id", "image", "file_name", "file"))
        )
        image = image_by_id.get(image_key, {})
        bbox = _first_mapping_value(annotation, ("bbox", "bbox_xywh", "box", "xywh"))
        if bbox is None:
            continue
        category_id = _first_mapping_value(
            annotation,
            ("category_id", "class_id", "class", "label"),
        )
        rows.append(
            {
                "time_s": _coco_time_s(annotation, image),
                "source": _coco_source(image),
                "track_id": _first_mapping_value(annotation, ("id", "annotation_id"))
                or f"annotation:{annotation_idx}",
                "bbox": bbox,
                "confidence": _first_mapping_value(
                    annotation,
                    ("score", "confidence", "probability"),
                ),
                "class_name": category_by_id.get(_coco_key(category_id), category_id),
                "depth_m": _first_mapping_value(
                    annotation,
                    ("depth_m", "depth", "range_m", "distance_m"),
                ),
                "image_file": _first_mapping_value(image, ("file_name", "file", "path")),
            }
        )
    return pd.DataFrame.from_records(rows)


def _coco_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _coco_category_map(categories: Any) -> dict[str, Any]:
    if not isinstance(categories, list):
        return {}
    out: dict[str, Any] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        category_id = _first_mapping_value(category, ("id", "category_id", "class_id"))
        name = _first_mapping_value(category, ("name", "class_name", "label", "category"))
        if category_id is not None and name is not None:
            out[_coco_key(category_id)] = name
    return out


def _coco_time_s(annotation: dict[Any, Any], image: dict[Any, Any]) -> float | None:
    for mapping in (annotation, image):
        timestamp = _stamp_to_seconds(
            _first_mapping_value(
                mapping,
                (
                    "time_s",
                    "timestamp_s",
                    "timestamp",
                    "time",
                    "t",
                    "stamp",
                ),
            )
        )
        if timestamp is not None:
            return timestamp
    file_name = _first_mapping_value(image, ("file_name", "file", "path"))
    if file_name is not None:
        return _timestamp_from_stem_or_none(Path(str(file_name)))
    return None


def _coco_source(image: dict[Any, Any]) -> Any | None:
    source = _first_mapping_value(image, ("source", "camera", "camera_id", "sensor"))
    if source is not None:
        return source
    file_name = _first_mapping_value(image, ("file_name", "file", "path"))
    if file_name is None:
        return None
    parent = Path(str(file_name)).parent
    return parent.name if parent.name else None


def _json_detection_records(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    if _looks_like_detection_column_map(payload) or _looks_like_detection_row(payload):
        return payload
    for key in (
        "camera_detections",
        "image_detections",
        "detections",
        "annotations",
        "boxes",
        "bboxes",
        "objects",
        "predictions",
        "results",
        "instances",
        "rows",
        "data",
    ):
        nested = _mapping_get_case_insensitive(payload, key)
        if nested is not None:
            return _json_detection_records(nested)
    return []


def _json_detection_records_to_frame(records: Any, *, path: Path | None = None) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, dict):
        if _looks_like_detection_column_map(records):
            return pd.DataFrame(records)
        if _looks_like_detection_row(records):
            return pd.DataFrame.from_records([_flatten_detection_record(records)])
    if isinstance(records, list):
        if not records:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in records):
            return pd.DataFrame.from_records(
                [_flatten_detection_record(item) for item in records]
            )
    label = str(path) if path is not None else "JSON payload"
    raise ValueError(f"camera detection JSON table {label} does not contain row objects")


def _flatten_detection_record(record: dict[Any, Any]) -> dict[Any, Any]:
    out = dict(record)
    header = _mapping_get_case_insensitive(out, "header")
    if isinstance(header, dict):
        _copy_stamp_time(out, _mapping_get_case_insensitive(header, "stamp"))
        frame_id = _mapping_get_case_insensitive(header, "frame_id")
        if frame_id is not None and not _has_any_key(
            out,
            ("source", "camera", "camera_id", "sensor"),
        ):
            out["source"] = frame_id
    if not _has_time_key(out):
        _copy_stamp_time(out, _mapping_get_case_insensitive(out, "stamp"))
    bbox = _mapping_get_case_insensitive(out, "bbox")
    if isinstance(bbox, dict):
        _copy_ros_bbox_fields(out, bbox)
    results = _mapping_get_case_insensitive(out, "results")
    if isinstance(results, list):
        _copy_detection_result_fields(out, results)
    return out


def _copy_stamp_time(out: dict[Any, Any], stamp: Any) -> None:
    if _has_time_key(out):
        return
    time_s = _stamp_to_seconds(stamp)
    if time_s is not None:
        out["time_s"] = time_s


def _stamp_to_seconds(stamp: Any) -> float | None:
    if isinstance(stamp, dict):
        nested = _mapping_get_case_insensitive(stamp, "stamp")
        if nested is not None:
            nested_time = _stamp_to_seconds(nested)
            if nested_time is not None:
                return nested_time
        seconds = _first_mapping_value(stamp, ("sec", "secs", "seconds"))
        nanoseconds = _first_mapping_value(
            stamp,
            ("nanosec", "nsec", "nsecs", "nanoseconds"),
        )
        if seconds is not None:
            try:
                return float(seconds) + (float(nanoseconds or 0.0) * 1.0e-9)
            except (TypeError, ValueError):
                return None
        numeric = _first_mapping_value(
            stamp,
            ("time_s", "timestamp_s", "timestamp", "stamp", "time"),
        )
        return _float_or_none(numeric)
    return _float_or_none(stamp)


def _copy_ros_bbox_fields(out: dict[Any, Any], bbox: dict[Any, Any]) -> None:
    center = _mapping_get_case_insensitive(bbox, "center")
    position = None
    if isinstance(center, dict):
        position = _mapping_get_case_insensitive(center, "position")
    center_mapping = position if isinstance(position, dict) else center
    if isinstance(center_mapping, dict):
        u_px = _first_mapping_value(center_mapping, ("x", "u", "cx", "center_x"))
        v_px = _first_mapping_value(center_mapping, ("y", "v", "cy", "center_y"))
        depth_m = _first_mapping_value(
            center_mapping,
            ("z", "depth", "depth_m", "range", "range_m", "distance", "distance_m"),
        )
        if depth_m is None and center_mapping is not center and isinstance(center, dict):
            depth_m = _first_mapping_value(
                center,
                ("z", "depth", "depth_m", "range", "range_m", "distance", "distance_m"),
            )
        _set_if_missing(
            out,
            "u_px",
            u_px,
            ("u", "pixel_x", "center_x", "bbox_center_x"),
        )
        _set_if_missing(
            out,
            "v_px",
            v_px,
            ("v", "pixel_y", "center_y", "bbox_center_y"),
        )
        _set_if_missing(
            out,
            "depth_m",
            depth_m,
            ("depth", "range", "range_m", "distance", "distance_m", "z_depth_m"),
        )
        width = _first_mapping_value(bbox, ("size_x", "width", "w"))
        height = _first_mapping_value(bbox, ("size_y", "height", "h"))
        try:
            u = float(u_px)
            v = float(v_px)
            half_width = float(width) / 2.0
            half_height = float(height) / 2.0
        except (TypeError, ValueError):
            return
        _set_if_missing(out, "x1", u - half_width, ("xmin", "bbox_x1", "left"))
        _set_if_missing(out, "y1", v - half_height, ("ymin", "bbox_y1", "top"))
        _set_if_missing(out, "x2", u + half_width, ("xmax", "bbox_x2", "right"))
        _set_if_missing(out, "y2", v + half_height, ("ymax", "bbox_y2", "bottom"))


def _copy_detection_result_fields(out: dict[Any, Any], results: list[Any]) -> None:
    best_score: float | None = None
    best_class: Any | None = None
    for result in results:
        if not isinstance(result, dict):
            continue
        hypothesis = _mapping_get_case_insensitive(result, "hypothesis")
        if not isinstance(hypothesis, dict):
            hypothesis = result
        score = _float_or_none(
            _first_mapping_value(hypothesis, ("score", "confidence", "probability"))
        )
        if score is None:
            score = _float_or_none(
                _first_mapping_value(result, ("score", "confidence", "probability"))
            )
        class_name = _first_mapping_value(
            hypothesis,
            ("class_id", "class_name", "class", "label", "category", "id"),
        )
        if class_name is None:
            class_name = _first_mapping_value(
                result,
                ("class_id", "class_name", "class", "label", "category"),
            )
        if best_score is None or (score is not None and score > best_score):
            best_score = score
            best_class = class_name
    _set_if_missing(out, "confidence", best_score, ("score", "probability"))
    _set_if_missing(out, "class_name", best_class, ("class", "label", "category"))


def _first_mapping_value(mapping: dict[Any, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        value = _mapping_get_case_insensitive(mapping, key)
        if value is not None:
            return value
    return None


def _set_if_missing(
    mapping: dict[Any, Any],
    key: str,
    value: Any,
    aliases: Iterable[str] = (),
) -> None:
    if value is None:
        return
    if not _has_any_key(mapping, (key, *tuple(aliases))):
        mapping[key] = value


def _has_time_key(mapping: dict[Any, Any]) -> bool:
    return _has_any_key(
        mapping,
        (
            "time_s",
            "timestamp",
            "timestamp_s",
            "timestamp_ns",
            "timestamp_ms",
            "sec",
            "secs",
            "nanosec",
        ),
    )


def _has_any_key(mapping: dict[Any, Any], keys: Iterable[str]) -> bool:
    present = {str(key).lower() for key in mapping}
    return any(key.lower() in present for key in keys)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    "center_x",
    "center_x_px",
    "bbox_center_x",
    "bbox_cx",
    "v_px",
    "v",
    "pixel_y",
    "center_v",
    "center_y",
    "center_y_px",
    "bbox_center_y",
    "bbox_cy",
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
    "depth",
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
