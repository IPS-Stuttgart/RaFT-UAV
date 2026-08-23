"""Compatibility fixes for the MMUAD radar-extraction audit.

The maintained implementation lives in the sibling ``radar_extraction_audit.py``
module. This package preserves the public import path while validating the voxel
size before the audit's intentionally tolerant per-frame error handling can hide
configuration errors. It also uses the production NumPy point-cloud loader so
NPZ array selection and resource handling agree with candidate extraction.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from raft_uav.mmuad.io import _numpy_array_from_export
from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "radar_extraction_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._radar_extraction_audit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load radar-extraction audit from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_BUILD_RADAR_EXTRACTION_AUDIT = _IMPL.build_radar_extraction_audit
_POINT_ARRAY_KEYS = (
    "points",
    "point_cloud",
    "pointcloud",
    "cloud",
    "lidar_points",
    "livox_points",
    "rows",
    "data",
)


def _validated_voxel_size_m(value: object) -> float:
    """Return one positive finite scalar voxel size."""

    normalized = optional_float(value)
    if normalized is None or normalized <= 0.0:
        raise ValueError("voxel_size_m must be a positive finite scalar")
    return normalized


def _load_numpy_array(path: Path) -> np.ndarray:
    """Load the same NPZ point array used by production candidate extraction."""

    path = Path(path)
    if path.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError(f"unsupported raw radar format {path.suffix!r}")
    return np.asarray(
        _numpy_array_from_export(path, preferred_keys=_POINT_ARRAY_KEYS)
    )


def build_radar_extraction_audit(
    sequence_root: Path,
    *,
    sequence_glob: str = "*",
    voxel_size_m: float = 0.75,
) -> Any:
    """Build the audit after rejecting malformed clustering configuration."""

    return _ORIGINAL_BUILD_RADAR_EXTRACTION_AUDIT(
        sequence_root,
        sequence_glob=sequence_glob,
        voxel_size_m=_validated_voxel_size_m(voxel_size_m),
    )


_IMPL._load_numpy_array = _load_numpy_array
_IMPL.build_radar_extraction_audit = build_radar_extraction_audit

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_POINT_ARRAY_KEYS"] = _POINT_ARRAY_KEYS
globals()["_validated_voxel_size_m"] = _validated_voxel_size_m
globals()["_load_numpy_array"] = _load_numpy_array
globals()["build_radar_extraction_audit"] = build_radar_extraction_audit

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
