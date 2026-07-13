"""Compatibility wrapper validating vertical-repair iteration counts.

The maintained implementation lives in the sibling ``track5_vertical_repair.py``
module. This package preserves the public import path while rejecting malformed
``iterations`` values instead of silently truncating or clamping them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_vertical_repair.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_vertical_repair_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load vertical-repair implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_REPAIR = _IMPL.repair_track5_vertical_spikes


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer without lossy or Boolean coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def repair_track5_vertical_spikes(
    submission,
    *,
    max_vertical_speed_mps: float = 20.0,
    max_neighbor_vertical_speed_mps: float = 10.0,
    max_vertical_residual_m: float = 15.0,
    max_horizontal_speed_mps: float | None = 80.0,
    iterations: int = 2,
):
    """Return vertically repaired estimates after validating ``iterations``."""

    validated_iterations = _positive_integer(iterations, name="iterations")
    return _ORIGINAL_REPAIR(
        submission,
        max_vertical_speed_mps=max_vertical_speed_mps,
        max_neighbor_vertical_speed_mps=max_neighbor_vertical_speed_mps,
        max_vertical_residual_m=max_vertical_residual_m,
        max_horizontal_speed_mps=max_horizontal_speed_mps,
        iterations=validated_iterations,
    )


_IMPL.repair_track5_vertical_spikes = repair_track5_vertical_spikes

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_positive_integer"] = _positive_integer
globals()["repair_track5_vertical_spikes"] = repair_track5_vertical_spikes

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
