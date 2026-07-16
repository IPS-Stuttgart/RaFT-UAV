"""Compatibility fixes for Track 5 scorecard inputs.

The maintained implementation lives in the sibling ``track5_scorecard.py``
module. This package preserves the public import path while retaining opaque
sequence identifiers and normalizing serialized Boolean diagnostics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_scorecard.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_scorecard_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 scorecard implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_SEQUENCE_IDENTIFIER_DTYPES = {
    "sequence_id": "string",
    "sequence": "string",
    "Sequence": "string",
}


def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    """Load optional scorecard diagnostics while preserving opaque IDs."""

    if path is None:
        return None

    columns = pd.read_csv(path, nrows=0).columns
    identifier_dtypes = {
        name: dtype
        for name, dtype in _SEQUENCE_IDENTIFIER_DTYPES.items()
        if name in columns
    }
    return pd.read_csv(path, dtype=identifier_dtypes)


def _bool_series(values: Any) -> pd.Series:
    """Normalize Boolean diagnostics, including CSV-style ``1.0`` / ``0.0``."""

    if values is None:
        return pd.Series(dtype=bool)
    series = pd.Series(values)
    if series.empty:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    normalized = pd.Series(False, index=series.index, dtype=bool)
    normalized.loc[numeric_mask] = numeric.loc[numeric_mask].eq(1.0)

    text = series.fillna(False).astype(str).str.strip().str.lower()
    normalized.loc[~numeric_mask] = text.loc[~numeric_mask].isin(
        {"1", "true", "t", "yes", "y"}
    )
    return normalized


_IMPL._load_optional_csv = _load_optional_csv
_IMPL._bool_series = _bool_series

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_load_optional_csv"] = _load_optional_csv
globals()["_bool_series"] = _bool_series

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
