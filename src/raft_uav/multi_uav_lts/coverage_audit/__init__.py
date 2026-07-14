"""Compatibility package validating Multi-UAV LTS coverage thresholds.

The maintained implementation lives in the sibling ``coverage_audit.py``
module. This package preserves the public import path while rejecting malformed
coverage thresholds that would otherwise disable the low-coverage readiness
check through IEEE non-finite comparison semantics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "coverage_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.multi_uav_lts._coverage_audit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load coverage audit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _normalized_min_detection_fraction(value: Any) -> float | None:
    """Return one finite scalar coverage fraction in the closed unit interval."""

    if value is None:
        return None
    message = (
        "--min-detection-frame-fraction must be a finite number in [0, 1]"
    )
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(message)
    return normalized


_IMPL._normalized_min_detection_fraction = _normalized_min_detection_fraction

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalized_min_detection_fraction"] = (
    _normalized_min_detection_fraction
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
