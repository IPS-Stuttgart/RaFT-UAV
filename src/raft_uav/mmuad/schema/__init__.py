"""Compatibility fixes for MMUAD schema normalization.

The maintained implementation lives in the sibling ``schema.py`` module. This
package preserves the public import path while ensuring that values nested in
NumPy arrays are normalized recursively before JSON serialization and that the
``NaT`` missing-value sentinel is not retained as a literal sequence identifier.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "schema.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._schema_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load MMUAD schema implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_LOAD_JSONABLE = _IMPL.load_jsonable
_ORIGINAL_NORMALIZE_SEQUENCE_ID_VALUES = _IMPL._normalize_sequence_id_values


def load_jsonable(value: Any) -> Any:
    """Return recursively normalized JSON-safe values, including array elements."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    return _ORIGINAL_LOAD_JSONABLE(value)


def _normalize_sequence_id_values(
    values: pd.Series,
    *,
    default_sequence_id: str,
) -> pd.Series:
    """Normalize sequence ids, including serialized ``NaT`` sentinels."""

    normalized = _ORIGINAL_NORMALIZE_SEQUENCE_ID_VALUES(
        values,
        default_sequence_id=default_sequence_id,
    )
    missing_nat = normalized.astype(str).str.strip().str.casefold().eq("nat")
    return normalized.where(~missing_nat, str(default_sequence_id))


_IMPL.load_jsonable = load_jsonable
_IMPL._normalize_sequence_id_values = _normalize_sequence_id_values

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["load_jsonable"] = load_jsonable
globals()["_normalize_sequence_id_values"] = _normalize_sequence_id_values

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
