"""Compatibility wrapper for robust pandas-derived tracklet summaries.

The maintained implementation lives in the sibling ``pyrecest_tracklet_graph.py``
module. This package preserves the public import path while making its summary
adapter treat pandas missing scalars as absent values instead of evaluating them
in a boolean context.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "pyrecest_tracklet_graph.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._pyrecest_tracklet_graph_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load PyRecEst tracklet-graph adapter from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_FORTEM_TRACKLET_FROM_SUMMARY = _IMPL.fortem_tracklet_from_summary
_ORIGINAL_JSONABLE = _IMPL._jsonable


def _is_missing_scalar(value: Any) -> bool:
    """Return whether one mapping value should be treated as absent."""

    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _first_present(row: Mapping[str, Any], *names: str) -> Any:
    """Return the first non-missing alias without coercing ``pd.NA`` to bool."""

    for name in names:
        if name in row and not _is_missing_scalar(row[name]):
            return row[name]
    raise KeyError(f"row must contain one of {names!r}")


def _optional(row: Mapping[str, Any], name: str, *, default: Any = None) -> Any:
    """Return a scalar mapping value or ``default`` when it is missing."""

    value = row.get(name, default)
    return default if _is_missing_scalar(value) else value


def _jsonable(value: Any) -> Any:
    """Convert pandas missing scalars to JSON-safe ``None`` values."""

    if _is_missing_scalar(value):
        return None
    return _ORIGINAL_JSONABLE(value)


def fortem_tracklet_from_summary(
    row: Mapping[str, Any],
    *,
    cost_key: str = "score",
):
    """Convert a summary row while safely defaulting a missing node cost."""

    normalized = dict(row)
    if _is_missing_scalar(normalized.get(cost_key)):
        normalized[cost_key] = 0.0
    return _ORIGINAL_FORTEM_TRACKLET_FROM_SUMMARY(
        normalized,
        cost_key=cost_key,
    )


_IMPL._first_present = _first_present
_IMPL._optional = _optional
_IMPL._jsonable = _jsonable
_IMPL.fortem_tracklet_from_summary = fortem_tracklet_from_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_is_missing_scalar"] = _is_missing_scalar
globals()["_first_present"] = _first_present
globals()["_optional"] = _optional
globals()["_jsonable"] = _jsonable
globals()["fortem_tracklet_from_summary"] = fortem_tracklet_from_summary
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
