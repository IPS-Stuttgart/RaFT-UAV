"""Compatibility wrapper validating candidate-classification thresholds.

The maintained implementation lives in the sibling ``classification.py``
module. This package preserves the public import path while ensuring malformed
candidate confidence thresholds cannot silently produce all-default class maps.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "classification.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._classification_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load MMUAD classification implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_INFER_SEQUENCE_CLASS_MAP = _IMPL.infer_sequence_class_map_from_candidates


def _normalize_min_confidence(value: Any) -> float:
    """Return a finite scalar confidence threshold or raise a stable error."""

    message = "min_confidence must be a finite real scalar"
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
    if not np.isfinite(numeric):
        raise ValueError(message)
    return numeric


def infer_sequence_class_map_from_candidates(
    candidates,
    *,
    min_confidence: float = 0.0,
    default_class: str = "unknown",
):
    """Infer classes after validating the candidate-confidence threshold."""

    return _ORIGINAL_INFER_SEQUENCE_CLASS_MAP(
        candidates,
        min_confidence=_normalize_min_confidence(min_confidence),
        default_class=default_class,
    )


_IMPL.infer_sequence_class_map_from_candidates = infer_sequence_class_map_from_candidates

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["infer_sequence_class_map_from_candidates"] = infer_sequence_class_map_from_candidates
globals()["_normalize_min_confidence"] = _normalize_min_confidence

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
