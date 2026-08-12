"""Compatibility package validating MMUAD point-cloud controls and metadata.

The maintained I/O compatibility layer lives in the sibling ``io.py`` module.
This package preserves that public import path while rejecting malformed integer
controls before dynamic background removal can silently clamp or truncate them.
It also rejects unsupported PCD field widths and malformed PCD COUNT layouts
before binary records can be decoded with shifted offsets and corrupted
coordinates, and rejects ambiguous raw BIN row widths unless the path identifies
a sensor with a documented export layout.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "io.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._io_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load MMUAD I/O compatibility layer from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

_ORIGINAL_DYNAMIC_POINT_RESIDUALS = _LEGACY._dynamic_point_residuals
_ORIGINAL_PARSE_PCD_HEADER = _LEGACY._impl._parse_pcd_header
_BINARY_POINT_COLUMNS_ENV = "RAFT_UAV_BINARY_POINT_COLUMNS"
_PCD_NUMPY_DTYPES: dict[str, dict[int, str]] = {
    "F": {4: "<f4", 8: "<f8"},
    "I": {1: "<i1", 2: "<i2", 4: "<i4", 8: "<i8"},
    "U": {1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"},
}


def _exact_integer_control(value: Any, *, name: str, minimum: int) -> int:
    """Return an exact integer scalar satisfying ``minimum``."""

    qualifier = "positive" if minimum == 1 else "non-negative"
    message = f"{name} must be a {qualifier} integer"
    normalized = optional_int(value)
    if normalized is None or normalized < minimum:
        raise ValueError(message)
    return normalized


def _validate_pcd_counts(header: dict[str, object]) -> dict[str, object]:
    """Reject explicit PCD COUNT metadata that cannot describe ``FIELDS``."""

    if "count" not in header:
        return header
    fields = list(header.get("fields", []))
    counts = list(header.get("count", []))
    if len(counts) != len(fields):
        raise ValueError("PCD header has inconsistent FIELDS/COUNT lengths")
    normalized_counts: list[int] = []
    for field, count in zip(fields, counts, strict=True):
        normalized = optional_int(count)
        if normalized is None or normalized < 1:
            raise ValueError(f"PCD COUNT for field {field!r} must be a positive integer")
        normalized_counts.append(normalized)
    validated = dict(header)
    validated["count"] = normalized_counts
    return validated


def _parse_pcd_header(header_text: str) -> dict[str, object]:
    """Parse PCD metadata and validate explicit field multiplicities."""

    return _validate_pcd_counts(_ORIGINAL_PARSE_PCD_HEADER(header_text))


def _pcd_numpy_dtype(*, size: int, type_code: str) -> str:
    """Return an exact NumPy dtype for a supported PCD TYPE/SIZE pair."""

    code = str(type_code).strip().upper()
    if code not in _PCD_NUMPY_DTYPES:
        raise ValueError(f"unsupported PCD type code: {type_code!r}")
    normalized_size = optional_int(size)
    supported = _PCD_NUMPY_DTYPES[code]
    if normalized_size not in supported:
        raise ValueError(
            f"unsupported PCD SIZE {size!r} for TYPE {code!r}; "
            f"supported sizes are {sorted(supported)}"
        )
    return supported[normalized_size]


def _binary_point_columns_from_filename(path: Path) -> int | None:
    """Return an XYZ/XYZI width encoded as a standalone filename token."""

    tokens = set(filter(None, re.split(r"[^a-z0-9]+", Path(path).name.casefold())))
    hints = {3 for token in tokens if token == "xyz"}
    hints.update(4 for token in tokens if token == "xyzi")
    if len(hints) > 1:
        raise ValueError(
            f"binary point-cloud filename {path} contains conflicting XYZ and XYZI hints"
        )
    return next(iter(hints), None)


def _binary_point_columns_from_environment() -> int | None:
    """Return an optional process-wide BIN row width override."""

    raw = os.environ.get(_BINARY_POINT_COLUMNS_ENV)
    if raw is None or not raw.strip():
        return None
    aliases = {"3": 3, "xyz": 3, "4": 4, "xyzi": 4}
    normalized = aliases.get(raw.strip().casefold())
    if normalized is None:
        raise ValueError(
            f"{_BINARY_POINT_COLUMNS_ENV} must be one of 3, xyz, 4, or xyzi"
        )
    return normalized


def _binary_point_columns_from_sensor_path(path: Path) -> int | None:
    """Return the documented row width implied by a known sensor path."""

    tokens: set[str] = set()
    for part in Path(path).parts:
        tokens.update(filter(None, re.split(r"[^a-z0-9]+", part.casefold())))
    # Livox raw exports store x, y, z, and reflectivity as four float32 values.
    return 4 if "livox" in tokens else None


def _binary_point_column_count(path: Path, value_count: int) -> int:
    """Resolve a deterministic BIN row width without guessing generic data."""

    filename_hint = _binary_point_columns_from_filename(path)
    environment_hint = _binary_point_columns_from_environment()
    if (
        filename_hint is not None
        and environment_hint is not None
        and filename_hint != environment_hint
    ):
        raise ValueError(
            f"binary point-cloud width hint for {path} conflicts with "
            f"{_BINARY_POINT_COLUMNS_ENV}={environment_hint}"
        )
    hinted = filename_hint if filename_hint is not None else environment_hint
    if hinted is None:
        hinted = _binary_point_columns_from_sensor_path(path)
    if hinted is not None:
        if value_count % hinted != 0:
            raise ValueError(
                f"binary point cloud {path} contains {value_count} float32 values, "
                f"which is not divisible into {hinted}-column rows"
            )
        return hinted

    possible = [columns for columns in (3, 4) if value_count % columns == 0]
    if len(possible) == 1:
        return possible[0]
    if len(possible) == 2:
        raise ValueError(
            f"binary point cloud {path} contains {value_count} float32 values, which "
            "is ambiguous between XYZ and XYZI rows; add an '.xyz.' or '.xyzi.' "
            f"filename token, or set {_BINARY_POINT_COLUMNS_ENV}=3 or 4"
        )
    raise ValueError(
        f"binary point cloud {path} must contain float32 XYZ or XYZI rows"
    )


def _read_binary_point_cloud(path: Path) -> pd.DataFrame:
    """Read little-endian float32 XYZ/XYZI rows without ambiguous reshaping."""

    payload = _LEGACY.read_binary_export(path)
    if len(payload) % np.dtype("<f4").itemsize != 0:
        raise ValueError(
            f"binary point cloud {path} byte length is not a whole number of float32 values"
        )
    raw = np.frombuffer(payload, dtype="<f4")
    if raw.size < 3:
        raise ValueError(f"binary point cloud {path} contains fewer than 3 float32 values")
    columns = _binary_point_column_count(path, int(raw.size))
    rows = raw.reshape(-1, columns)
    frame = pd.DataFrame(
        {"x_m": rows[:, 0], "y_m": rows[:, 1], "z_m": rows[:, 2]}
    )
    return _LEGACY._normalize_point_frame(frame, path=path)


def _dynamic_point_residuals(
    points,
    *,
    voxel_size_m: float,
    min_frame_fraction: float,
    min_frames: int,
    neighbor_radius_voxels: int,
):
    """Remove persistent voxels after validating exact integer controls."""

    normalized_min_frames = _exact_integer_control(
        min_frames,
        name="--dynamic-background-min-frames",
        minimum=1,
    )
    normalized_radius = _exact_integer_control(
        neighbor_radius_voxels,
        name="--dynamic-background-neighbor-radius-voxels",
        minimum=0,
    )
    return _ORIGINAL_DYNAMIC_POINT_RESIDUALS(
        points,
        voxel_size_m=voxel_size_m,
        min_frame_fraction=min_frame_fraction,
        min_frames=normalized_min_frames,
        neighbor_radius_voxels=normalized_radius,
    )


# The exported point-cloud helpers are implemented in ``_io_impl`` and resolve
# these functions through that module's globals, so patch both compatibility layers.
_LEGACY._dynamic_point_residuals = _dynamic_point_residuals
_LEGACY._impl._dynamic_point_residuals = _dynamic_point_residuals
_LEGACY._parse_pcd_header = _parse_pcd_header
_LEGACY._impl._parse_pcd_header = _parse_pcd_header
_LEGACY._pcd_numpy_dtype = _pcd_numpy_dtype
_LEGACY._impl._pcd_numpy_dtype = _pcd_numpy_dtype
_LEGACY._read_binary_point_cloud = _read_binary_point_cloud
_LEGACY._impl._read_binary_point_cloud = _read_binary_point_cloud

for _name in dir(_LEGACY):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_LEGACY, _name)
globals()["_exact_integer_control"] = _exact_integer_control
globals()["_validate_pcd_counts"] = _validate_pcd_counts
globals()["_parse_pcd_header"] = _parse_pcd_header
globals()["_pcd_numpy_dtype"] = _pcd_numpy_dtype
globals()["_read_binary_point_cloud"] = _read_binary_point_cloud
globals()["_dynamic_point_residuals"] = _dynamic_point_residuals

__doc__ = _LEGACY.__doc__
__all__ = sorted(
    {
        *[
            name
            for name in dir(_LEGACY)
            if not (name.startswith("__") and name.endswith("__"))
        ],
        "_dynamic_point_residuals",
        "_parse_pcd_header",
        "_pcd_numpy_dtype",
        "_read_binary_point_cloud",
        "_validate_pcd_counts",
    }
)
