"""Compatibility wrapper validating assignment-summary Boolean flags.

The maintained implementation lives in the sibling
``candidate_assignment_branch_summary.py`` module. This package preserves the
public import path while rejecting malformed serialized Boolean diagnostics
instead of silently counting arbitrary nonzero values as true or unknown text
as false.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_assignment_branch_summary.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_assignment_branch_summary_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load assignment branch summary implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_NORMALIZED_FRAME_ROWS = _IMPL._normalized_frame_rows
_TRUE_BOOLEAN_TEXT = frozenset({"true", "t", "yes", "y", "1", "1.0"})
_FALSE_BOOLEAN_TEXT = frozenset(
    {
        "false",
        "f",
        "no",
        "n",
        "0",
        "0.0",
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }
)
_BOOLEAN_COLUMNS = ("dominant_is_oracle", "oracle_in_topk_by_weight")


def _bool_series(values: Any, *, column_name: str = "Boolean column") -> pd.Series:
    """Parse native and serialized Boolean values without truthiness coercion."""

    series = pd.Series(values, copy=False)
    if series.empty:
        return pd.Series(index=series.index, dtype=bool)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean").fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    truthy = (text.isin(_TRUE_BOOLEAN_TEXT) | numeric.eq(1.0)).fillna(False)
    falsy = (
        series.isna() | text.isin(_FALSE_BOOLEAN_TEXT) | numeric.eq(0.0)
    ).fillna(False)
    invalid = ~(truthy | falsy)
    if bool(invalid.any()):
        invalid_indices = series.index[invalid.to_numpy(dtype=bool)].tolist()
        invalid_values = series.loc[invalid].tolist()
        raise ValueError(
            f"{column_name} contains invalid Boolean values at rows "
            f"{invalid_indices}: {invalid_values}"
        )
    return truthy.astype(bool)


def _normalized_frame_rows(frame_rows: pd.DataFrame) -> pd.DataFrame:
    """Validate persisted Boolean diagnostics before legacy normalization."""

    rows = pd.DataFrame(frame_rows).copy()
    if rows.empty:
        return _ORIGINAL_NORMALIZED_FRAME_ROWS(rows)
    for column in _BOOLEAN_COLUMNS:
        if column in rows.columns:
            rows[column] = _bool_series(rows[column], column_name=column)
    return _ORIGINAL_NORMALIZED_FRAME_ROWS(rows)


_IMPL._bool_series = _bool_series
_IMPL._normalized_frame_rows = _normalized_frame_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_TRUE_BOOLEAN_TEXT"] = _TRUE_BOOLEAN_TEXT
globals()["_FALSE_BOOLEAN_TEXT"] = _FALSE_BOOLEAN_TEXT
globals()["_BOOLEAN_COLUMNS"] = _BOOLEAN_COLUMNS
globals()["_bool_series"] = _bool_series
globals()["_normalized_frame_rows"] = _normalized_frame_rows
