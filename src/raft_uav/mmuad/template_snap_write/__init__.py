"""Compatibility fixes for Track 5 template-snap output manifests.

The maintained implementation lives in the sibling ``template_snap_write.py``
module. This package preserves the public import path while rejecting malformed
Boolean diagnostics instead of silently changing manifest row counts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "template_snap_write.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._template_snap_write_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load template-snap writer implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_TRUE_BOOL_TEXT = frozenset({"1", "1.0", "true", "t", "yes", "y"})
_FALSE_BOOL_TEXT = frozenset(
    {
        "",
        "0",
        "0.0",
        "false",
        "f",
        "no",
        "n",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }
)


def _unwrap_boolean_scalar(value: Any, *, field: str, row_index: Any) -> Any:
    """Recursively unwrap zero-dimensional arrays and reject unsafe containers."""

    seen: set[int] = set()
    while isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(
                f"{field} contains a non-scalar Boolean value at row {row_index!r}"
            )
        marker = id(value)
        if marker in seen:
            raise ValueError(
                f"{field} contains a cyclic Boolean value at row {row_index!r}"
            )
        seen.add(marker)
        value = value.item()
    return value


def _boolean_value(value: Any, *, field: str, row_index: Any) -> bool:
    """Normalize one diagnostic flag without lossy numeric coercion."""

    if value is None or value is pd.NA or np.ma.is_masked(value):
        return False
    value = _unwrap_boolean_scalar(value, field=field, row_index=row_index)
    if value is None or value is pd.NA or np.ma.is_masked(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(
            f"{field} contains an invalid Boolean value at row {row_index!r}: {value!r}"
        )
    if isinstance(value, str):
        token = value.strip().casefold()
        if token in _TRUE_BOOL_TEXT:
            return True
        if token in _FALSE_BOOL_TEXT:
            return False
        raise ValueError(
            f"{field} contains an invalid Boolean value at row {row_index!r}: {value!r}"
        )

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return False

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} contains an invalid Boolean value at row {row_index!r}: {value!r}"
        ) from exc
    if not np.isfinite(numeric) or numeric not in {0.0, 1.0}:
        raise ValueError(
            f"{field} contains an invalid Boolean value at row {row_index!r}: {value!r}"
        )
    return numeric == 1.0


def _bool_column(rows: pd.DataFrame, column: str) -> pd.Series:
    """Normalize native and serialized Boolean diagnostics strictly."""

    if column not in rows.columns:
        return pd.Series(False, index=rows.index, dtype=bool)
    values = pd.Series(rows[column], index=rows.index, copy=False)
    return pd.Series(
        (
            _boolean_value(value, field=column, row_index=index)
            for index, value in values.items()
        ),
        index=values.index,
        dtype=bool,
    )


_IMPL._unwrap_boolean_scalar = _unwrap_boolean_scalar
_IMPL._boolean_value = _boolean_value
_IMPL._bool_column = _bool_column

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_unwrap_boolean_scalar"] = _unwrap_boolean_scalar
globals()["_boolean_value"] = _boolean_value
globals()["_bool_column"] = _bool_column

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
