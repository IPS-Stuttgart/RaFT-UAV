"""Compatibility wrapper validating acceleration-repair iteration counts.

The maintained implementation lives in the sibling
``track5_acceleration_limit.py`` module. This package preserves the public import
path while preventing malformed iteration controls from being silently coerced.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_acceleration_limit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_acceleration_limit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load acceleration-limit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_REPAIR = _IMPL.repair_track5_acceleration_kinks


def _positive_integer(value: object, *, name: str) -> int:
    """Return ``value`` as an integer, rejecting lossy or Boolean coercions."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def repair_track5_acceleration_kinks(
    submission,
    *,
    max_acceleration_mps2: float = 20.0,
    max_direct_speed_mps: float = 80.0,
    min_interpolation_residual_m: float = 1.0,
    iterations: int = 2,
    repair_blend: float = 1.0,
):
    """Return acceleration-limited estimates after validating ``iterations``."""

    validated_iterations = _positive_integer(iterations, name="iterations")
    return _ORIGINAL_REPAIR(
        submission,
        max_acceleration_mps2=max_acceleration_mps2,
        max_direct_speed_mps=max_direct_speed_mps,
        min_interpolation_residual_m=min_interpolation_residual_m,
        iterations=validated_iterations,
        repair_blend=repair_blend,
    )


_IMPL.repair_track5_acceleration_kinks = repair_track5_acceleration_kinks

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["repair_track5_acceleration_kinks"] = repair_track5_acceleration_kinks

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
