"""Compatibility validation for MMUAD trajectory completion controls.

The maintained implementation lives in the sibling ``completion.py`` module.
This package preserves the public import path while rejecting malformed scalar
controls before completion or summary generation starts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from raft_uav.numeric import optional_int

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

_ORIGINAL_COMPLETION_SUMMARY = _IMPL.completion_summary


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


def _normalize_requested_count(value: object, *, completed_count: int) -> int:
    """Return an exact count consistent with the completed result rows."""

    requested = optional_int(value)
    if requested is None or requested < 0:
        raise ValueError("requested_count must be a non-negative integer")
    if requested < completed_count:
        raise ValueError(
            f"requested_count ({requested}) cannot be smaller than "
            f"completed_count ({completed_count})"
        )
    return requested


def completion_summary(result, *, requested_count: object | None = None):
    """Summarize completion after validating an explicit requested count."""

    normalized_requested_count = None
    if requested_count is not None:
        normalized_requested_count = _normalize_requested_count(
            requested_count,
            completed_count=len(result.rows),
        )
    return _ORIGINAL_COMPLETION_SUMMARY(
        result,
        requested_count=normalized_requested_count,
    )


_IMPL._normalize_max_interpolation_gap_s = _normalize_max_interpolation_gap_s
_IMPL.completion_summary = completion_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_max_interpolation_gap_s"] = _normalize_max_interpolation_gap_s
globals()["_normalize_requested_count"] = _normalize_requested_count
globals()["completion_summary"] = completion_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
