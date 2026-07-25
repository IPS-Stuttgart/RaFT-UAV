"""Compatibility fix for MMUAD Track 5 classification label validation.

The maintained implementation lives in the sibling ``classification_audit.py``
module. This package preserves the public import path while requiring every
classification row to contain a valid official class identifier.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "classification_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._classification_audit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load classification audit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _valid_class_series(values: pd.Series) -> bool:
    """Return whether every row contains a valid official class identifier."""

    normalized = pd.Series(values, copy=False)
    if normalized.empty or bool(normalized.isna().any()):
        return False
    return all(_IMPL._valid_class_id(value) for value in normalized.tolist())


_IMPL._valid_class_series = _valid_class_series

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_valid_class_series"] = _valid_class_series

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
