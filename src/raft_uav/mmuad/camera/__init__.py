"""Compatibility fixes for MMUAD camera loading and model lookup.

The maintained implementation lives in the sibling ``camera.py`` module. This
package preserves the public import path while selecting specific camera models
and correctly reading gzip-compressed YOLO label exports.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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


def _model_for_source(models, source):
    """Return the exact or longest one-way source-prefix camera model."""

    source_key = str(source).strip().lower()
    normalized = [
        (str(key).strip().lower(), model)
        for key, model in models.items()
    ]
    for key, model in normalized:
        if source_key == key:
            return model
    if len(models) == 1:
        return next(iter(models.values()))
    matches = [
        (len(key), model)
        for key, model in normalized
        if key and source_key.startswith(key)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


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
globals()["_model_for_source"] = _model_for_source
globals()["_export_stem"] = _export_stem
globals()["_same_stem_image_path"] = _same_stem_image_path
globals()["_looks_like_yolo_label_file"] = _looks_like_yolo_label_file
globals()["_read_yolo_label_table"] = _read_yolo_label_table

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
