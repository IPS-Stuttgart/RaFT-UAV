"""Compatibility wrapper validating Track 5 estimate sequence-gate weights.

The maintained implementation lives in the sibling
``track5_estimate_sequence_gate.py`` module. This package preserves the public
import path while rejecting Boolean, complex, non-scalar, masked, and non-finite
blend weights before they can silently select a trajectory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_estimate_sequence_gate.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_sequence_gate_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load estimate sequence-gate implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _validate_weight(value: Any, *, name: str) -> float:
    """Return one finite real scalar blend weight in the closed unit interval."""

    message = f"{name} must be a finite real scalar in [0, 1]"
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
    try:
        weight = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{message}: {value!r}") from exc
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"{message}: {value!r}")
    return weight


_IMPL._validate_weight = _validate_weight

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_weight"] = _validate_weight

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
