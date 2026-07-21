"""Compatibility validation for MMUAD trajectory completion controls.

The maintained implementation lives in the sibling ``completion.py`` module.
This package preserves the public import path while rejecting Boolean, complex,
masked, and non-scalar interpolation-gap values before completion starts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "completion.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._completion_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load completion implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _normalize_max_interpolation_gap_s(value: object) -> float:
    """Return a finite non-negative real scalar interpolation gap."""

    message = "max_interpolation_gap_s must be a finite non-negative number"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0 or array.dtype.kind in {"b", "c"}:
        raise ValueError(message)
    try:
        gap = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError(message)
    return gap


_IMPL._normalize_max_interpolation_gap_s = _normalize_max_interpolation_gap_s

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_max_interpolation_gap_s"] = _normalize_max_interpolation_gap_s

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
