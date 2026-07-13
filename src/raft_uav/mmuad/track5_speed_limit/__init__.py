"""Compatibility wrapper with strict Track 5 speed-limit iteration validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_speed_limit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_speed_limit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 speed-limit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_PROJECT_TRACK5_SPEED_LIMIT = _IMPL.project_track5_speed_limit


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def project_track5_speed_limit(
    submission,
    *,
    max_speed_mps: float = 60.0,
    iterations: int = 2,
    anchor_blend: float = 0.0,
):
    """Project a submission after validating the iteration count exactly."""

    normalized_iterations = _positive_integer(iterations, name="iterations")
    return _ORIGINAL_PROJECT_TRACK5_SPEED_LIMIT(
        submission,
        max_speed_mps=max_speed_mps,
        iterations=normalized_iterations,
        anchor_blend=anchor_blend,
    )


_IMPL.project_track5_speed_limit = project_track5_speed_limit

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["project_track5_speed_limit"] = project_track5_speed_limit

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
