"""Compatibility fixes for candidate-oracle block diagnostics.

The maintained implementation lives in the sibling ``candidate_oracle_blocks.py``
module. This package preserves the public import path while parsing persisted
Boolean top-K diagnostics explicitly instead of silently accepting malformed
values.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_blocks.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_blocks_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate-oracle block implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

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


def _to_bool_series(values: Any) -> pd.Series:
    """Parse native and serialized Boolean flags without silent coercion."""

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
        invalid_indices = invalid[invalid].index.tolist()
        invalid_values = series.loc[invalid_indices].tolist()
        raise ValueError(
            "oracle top-K flags contain invalid Boolean values at rows "
            f"{invalid_indices}: {invalid_values}"
        )
    return truthy.astype(bool)


_IMPL._to_bool_series = _to_bool_series

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_to_bool_series"] = _to_bool_series

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
