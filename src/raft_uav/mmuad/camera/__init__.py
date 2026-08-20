"""Compatibility fixes for MMUAD camera loading and model lookup.

The maintained implementation lives in the sibling ``camera.py`` module. This
package preserves the public import path while validating pinhole intrinsics and
camera time offsets, selecting specific camera models, rejecting ambiguous
detection columns, and correctly reading gzip-compressed YOLO label exports.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from raft_uav.mmuad.io import read_text_export

_IMPL_PATH = Path(__file__).resolve().parent.parent / "camera.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._camera_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load camera implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_LOAD_CAMERA_MODELS = _IMPL.load_camera_models
_ORIGINAL_INTRINSICS_FROM_CAMERA_ENTRY = _IMPL._intrinsics_from_camera_entry
_ORIGINAL_BACKPROJECT_PIXEL_TO_CAMERA_XYZ = _IMPL.backproject_pixel_to_camera_xyz
_ORIGINAL_NORMALIZE_CAMERA_DETECTION_COLUMNS = _IMPL._normalize_camera_detection_columns


def _validated_camera_intrinsics(intrinsics):
    """Return normalized finite pinhole intrinsics with positive focal lengths."""

    error = (
        "camera intrinsics must contain finite real scalars with "
        "fx > 0 and fy > 0"
    )
    values: dict[str, float] = {}
    for name in ("fx", "fy", "cx", "cy"):
        try:
            value = getattr(intrinsics, name)
        except AttributeError as exc:
            raise ValueError(error) from exc
        if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
            raise ValueError(error)
        try:
            scalar = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(error) from exc
        if scalar.ndim != 0 or np.iscomplexobj(scalar):
            raise ValueError(error)
        try:
            number = float(scalar.item())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(error) from exc
        if not np.isfinite(number):
            raise ValueError(error)
        values[name] = number
    if values["fx"] <= 0.0 or values["fy"] <= 0.0:
        raise ValueError(error)
    return _IMPL.CameraIntrinsics(**values)


def _validated_camera_time_offset(value):
    """Return a finite real camera time offset without accepting pseudo-numbers."""

    error = "camera time_offset_s must be a finite real scalar"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        item = scalar.item()
        if isinstance(item, (bool, np.bool_)) or np.ma.is_masked(item):
            raise ValueError(error)
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number):
        raise ValueError(error)
    return number


def _validated_camera_model(model):
    """Return a camera model with validated intrinsics and time offset."""

    try:
        intrinsics = model.intrinsics
        time_offset_s = model.time_offset_s
    except AttributeError as exc:
        raise ValueError("camera model must provide intrinsics and time_offset_s") from exc
    return _IMPL.replace(
        model,
        intrinsics=_validated_camera_intrinsics(intrinsics),
        time_offset_s=_validated_camera_time_offset(time_offset_s),
    )


def _intrinsics_from_camera_entry(entry):
    """Load and validate one camera's pinhole intrinsics."""

    return _validated_camera_intrinsics(
        _ORIGINAL_INTRINSICS_FROM_CAMERA_ENTRY(entry)
    )


def load_camera_models(path):
    """Load camera models and reject malformed time offsets immediately."""

    return {
        key: _validated_camera_model(model)
        for key, model in _ORIGINAL_LOAD_CAMERA_MODELS(path).items()
    }


def backproject_pixel_to_camera_xyz(u_px, v_px, depth_m, intrinsics):
    """Back-project only after validating the supplied camera intrinsics."""

    return _ORIGINAL_BACKPROJECT_PIXEL_TO_CAMERA_XYZ(
        u_px,
        v_px,
        depth_m,
        _validated_camera_intrinsics(intrinsics),
    )


def _normalize_camera_detection_columns(frame):
    """Normalize padded headers and reject case-insensitive column collisions."""

    normalized = frame.copy()
    columns = [str(column).strip() for column in normalized.columns]
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(column.casefold(), []).append(column)
    ambiguous = sorted(
        {
            column
            for group in groups.values()
            if len(group) > 1
            for column in group
        },
        key=lambda column: (column.casefold(), column),
    )
    if ambiguous:
        names = ", ".join(repr(column) for column in ambiguous)
        raise ValueError(
            "camera detection table has ambiguous columns after trimming whitespace "
            f"and ignoring case: {names}"
        )
    normalized.columns = columns
    return _ORIGINAL_NORMALIZE_CAMERA_DETECTION_COLUMNS(normalized)


def _model_for_source(models, source):
    """Return the exact or longest one-way source-prefix camera model."""

    source_key = str(source).strip().lower()
    normalized = [
        (str(key).strip().lower(), model)
        for key, model in models.items()
    ]
    for key, model in normalized:
        if source_key == key:
            return _validated_camera_model(model)
    if len(models) == 1:
        return _validated_camera_model(next(iter(models.values())))
    matches = [
        (len(key), model)
        for key, model in normalized
        if key and source_key.startswith(key)
    ]
    if not matches:
        return None
    return _validated_camera_model(max(matches, key=lambda item: item[0])[1])


def _export_stem(path: Path) -> str:
    """Return the filename stem after removing transparent gzip compression."""

    logical_path = Path(path)
    if logical_path.suffix.lower() == ".gz":
        logical_path = logical_path.with_suffix("")
    return logical_path.stem


def _same_stem_image_path(path: Path) -> Path | None:
    """Find the image associated with a plain or gzip-compressed label file."""

    stem = _export_stem(path)
    directory = Path(path).parent
    for suffix in _IMPL.YOLO_IMAGE_SUFFIXES:
        for candidate in (
            directory / f"{stem}{suffix}",
            directory / f"{stem}{suffix.upper()}",
        ):
            if candidate.exists():
                return candidate
    return None


def _looks_like_yolo_label_file(path: Path) -> bool:
    """Detect YOLO rows after transparently decompressing text exports."""

    try:
        lines = read_text_export(Path(path), errors="ignore").splitlines()
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


def _read_yolo_label_table(path: Path):
    """Read YOLO rows from plain or gzip-compressed label files."""

    image_path = _same_stem_image_path(path)
    image_size = _IMPL._image_size_px(image_path) if image_path is not None else None
    rows: list[dict[str, Any]] = []
    time_s = _IMPL._timestamp_from_stem(path)
    for line_idx, line in enumerate(read_text_export(Path(path)).splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [float(part) for part in stripped.split()]
        if len(parts) not in {5, 6}:
            continue
        class_id, center_x, center_y, width, height = parts[:5]
        confidence = parts[5] if len(parts) == 6 else 1.0
        if _IMPL._looks_normalized_box(center_x, center_y, width, height):
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
                "class_name": (
                    str(int(class_id)) if float(class_id).is_integer() else str(class_id)
                ),
                "track_id": f"{_export_stem(path)}:{line_idx}",
                "image_file": str(image_path) if image_path is not None else "",
            }
        )
    return _IMPL.pd.DataFrame.from_records(
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


_IMPL._validated_camera_intrinsics = _validated_camera_intrinsics
_IMPL._validated_camera_time_offset = _validated_camera_time_offset
_IMPL._validated_camera_model = _validated_camera_model
_IMPL._intrinsics_from_camera_entry = _intrinsics_from_camera_entry
_IMPL.load_camera_models = load_camera_models
_IMPL.backproject_pixel_to_camera_xyz = backproject_pixel_to_camera_xyz
_IMPL._normalize_camera_detection_columns = _normalize_camera_detection_columns
_IMPL._model_for_source = _model_for_source
_IMPL._export_stem = _export_stem
_IMPL._same_stem_image_path = _same_stem_image_path
_IMPL._looks_like_yolo_label_file = _looks_like_yolo_label_file
_IMPL._read_yolo_label_table = _read_yolo_label_table

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_LOAD_CAMERA_MODELS"] = _ORIGINAL_LOAD_CAMERA_MODELS
globals()["_validated_camera_intrinsics"] = _validated_camera_intrinsics
globals()["_validated_camera_time_offset"] = _validated_camera_time_offset
globals()["_validated_camera_model"] = _validated_camera_model
globals()["_intrinsics_from_camera_entry"] = _intrinsics_from_camera_entry
globals()["load_camera_models"] = load_camera_models
globals()["backproject_pixel_to_camera_xyz"] = backproject_pixel_to_camera_xyz
globals()["_normalize_camera_detection_columns"] = _normalize_camera_detection_columns
globals()["_model_for_source"] = _model_for_source
globals()["_export_stem"] = _export_stem
globals()["_same_stem_image_path"] = _same_stem_image_path
globals()["_looks_like_yolo_label_file"] = _looks_like_yolo_label_file
globals()["_read_yolo_label_table"] = _read_yolo_label_table

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
