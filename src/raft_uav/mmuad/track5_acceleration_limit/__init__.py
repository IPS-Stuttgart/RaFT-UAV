"""Compatibility wrapper validating Track 5 acceleration-limit iterations.

The maintained implementation lives in the sibling ``track5_acceleration_limit.py``
module. This package preserves the public import path while rejecting malformed
iteration counts before the implementation can silently coerce them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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
_ORIGINAL_REPAIR_SEQUENCE = _IMPL._repair_sequence


def _normalize_iterations(value: Any) -> int:
    """Return a positive integer iteration count or raise a field-specific error."""

    message = "iterations must be a positive finite integer"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def repair_track5_acceleration_kinks(submission, **kwargs):
    """Validate ``iterations`` before running the legacy acceleration repair."""

    kwargs["iterations"] = _normalize_iterations(kwargs.get("iterations", 2))
    return _ORIGINAL_REPAIR(submission, **kwargs)


def _repair_sequence(group, **kwargs):
    """Validate private repair-loop iteration counts for direct callers."""

    kwargs["iterations"] = _normalize_iterations(kwargs["iterations"])
    return _ORIGINAL_REPAIR_SEQUENCE(group, **kwargs)


_IMPL.repair_track5_acceleration_kinks = repair_track5_acceleration_kinks
_IMPL._repair_sequence = _repair_sequence

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["repair_track5_acceleration_kinks"] = repair_track5_acceleration_kinks
globals()["_repair_sequence"] = _repair_sequence
globals()["_normalize_iterations"] = _normalize_iterations

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
