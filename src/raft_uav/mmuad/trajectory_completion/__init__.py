"""Compatibility wrapper for serialized trajectory-selection flags.

The maintained implementation lives in the sibling ``trajectory_completion.py``
module. This package preserves the public import path while normalizing
``selected_path_update`` values before smoothing selects its measurement rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "trajectory_completion.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._trajectory_completion_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load MMUAD trajectory-completion implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ESTIMATE_ROWS = _IMPL._estimate_rows


def _boolean_series(values: Any, index: pd.Index) -> pd.Series:
    """Parse boolean-like values without making false strings truthy."""

    series = pd.Series(values, index=index)
    if series.empty:
        return pd.Series(False, index=index, dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return numeric.ne(0.0)

    text = series.astype("string").str.strip().str.lower()
    truthy = text.isin({"1", "1.0", "true", "t", "yes", "y"})
    falsey = text.isin(
        {
            "0",
            "0.0",
            "false",
            "f",
            "no",
            "n",
            "",
            "none",
            "null",
            "nan",
            "<na>",
            "nat",
        }
    )
    numeric = pd.to_numeric(text, errors="coerce").fillna(0.0).ne(0.0)
    return (truthy | (~falsey & numeric)).fillna(False).astype(bool)


def _estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    """Normalize serialized selection flags before legacy row preparation."""

    rows = pd.DataFrame(estimates).copy()
    if "selected_path_update" in rows.columns:
        rows["selected_path_update"] = _boolean_series(
            rows["selected_path_update"],
            rows.index,
        )
    return _ORIGINAL_ESTIMATE_ROWS(rows)


_IMPL._boolean_series = _boolean_series
_IMPL._estimate_rows = _estimate_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_boolean_series"] = _boolean_series
globals()["_estimate_rows"] = _estimate_rows

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
