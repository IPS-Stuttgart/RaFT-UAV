"""Compatibility wrapper with strict MOT match-distance validation.

The maintained implementation lives in the sibling ``mot.py`` module. This
package preserves the public import path while rejecting malformed matching
thresholds before they can corrupt multi-object metrics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "mot.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._mot_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load MOT implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_COMPUTE_MULTI_OBJECT_METRICS = _IMPL.compute_multi_object_metrics


def _normalize_match_distance_m(value: Any) -> float:
    """Return a finite, non-negative scalar MOT matching threshold."""

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(
                "match_distance_m must be a finite non-negative scalar number"
            )
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(
            "match_distance_m must be a finite non-negative number, not a boolean"
        )
    try:
        distance = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "match_distance_m must be a finite non-negative number"
        ) from exc
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("match_distance_m must be a finite non-negative number")
    return distance


def compute_multi_object_metrics(
    estimates,
    truth,
    *,
    match_distance_m: float = 25.0,
):
    """Compute MOT metrics using a validated finite matching threshold."""

    distance = _normalize_match_distance_m(match_distance_m)
    return _ORIGINAL_COMPUTE_MULTI_OBJECT_METRICS(
        estimates,
        truth,
        match_distance_m=distance,
    )


_IMPL.compute_multi_object_metrics = compute_multi_object_metrics

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_match_distance_m"] = _normalize_match_distance_m
globals()["compute_multi_object_metrics"] = compute_multi_object_metrics

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
