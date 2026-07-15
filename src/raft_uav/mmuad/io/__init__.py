"""Compatibility package validating MMUAD dynamic point-extraction controls.

The maintained I/O compatibility layer lives in the sibling ``io.py`` module.
This package preserves that public import path while rejecting malformed integer
controls before dynamic background removal can silently clamp or truncate them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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


def _exact_integer_control(value: Any, *, name: str, minimum: int) -> int:
    """Return an exact integer scalar satisfying ``minimum``."""

    qualifier = "positive" if minimum == 1 else "non-negative"
    message = f"{name} must be a {qualifier} integer"
    normalized = optional_int(value)
    if normalized is None or normalized < minimum:
        raise ValueError(message)
    return normalized


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
# this function through that module's globals, so patch both compatibility layers.
_LEGACY._dynamic_point_residuals = _dynamic_point_residuals
_LEGACY._impl._dynamic_point_residuals = _dynamic_point_residuals

for _name in dir(_LEGACY):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_LEGACY, _name)
globals()["_exact_integer_control"] = _exact_integer_control
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
    }
)
