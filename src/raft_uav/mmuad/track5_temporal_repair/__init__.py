"""Compatibility wrapper validating temporal-repair iteration controls.

The maintained implementation lives in the sibling ``track5_temporal_repair.py``
module. This package preserves the public import path while rejecting zero,
negative, fractional, Boolean, masked, and non-scalar iteration values before a
trajectory can be modified.
"""

from __future__ import annotations

import importlib.util
from numbers import Integral
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_temporal_repair.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_temporal_repair_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load temporal-repair implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_REPAIR_TRACK5_TEMPORAL_SPIKES = _IMPL.repair_track5_temporal_spikes


def _validate_iterations(value: Any) -> int:
    """Return an exact positive integer iteration count."""

    message = "iterations must be an exact positive integer"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(f"{message}: {value!r}")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{message}: {value!r}") from exc
    if array.ndim != 0 or array.dtype.kind in {"b", "c"}:
        raise ValueError(f"{message}: {value!r}")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{message}: {value!r}")
    if isinstance(scalar, Integral):
        iterations = int(scalar)
    else:
        try:
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{message}: {value!r}") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{message}: {value!r}")
        iterations = int(numeric)
    if iterations <= 0:
        raise ValueError(f"{message}: {value!r}")
    return iterations


def repair_track5_temporal_spikes(
    submission: Any,
    *,
    max_speed_mps: float = 80.0,
    max_interpolation_residual_m: float = 25.0,
    iterations: Any = 2,
) -> tuple[Any, Any]:
    """Return repaired estimates after validating the requested pass count."""

    return _ORIGINAL_REPAIR_TRACK5_TEMPORAL_SPIKES(
        submission,
        max_speed_mps=max_speed_mps,
        max_interpolation_residual_m=max_interpolation_residual_m,
        iterations=_validate_iterations(iterations),
    )


_IMPL.repair_track5_temporal_spikes = repair_track5_temporal_spikes

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_iterations"] = _validate_iterations
globals()["repair_track5_temporal_spikes"] = repair_track5_temporal_spikes

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
