"""Compatibility fixes for Track 5 scorecard inputs.

The maintained implementation lives in the sibling ``track5_scorecard.py``
module. This package preserves the public import path while retaining opaque
sequence identifiers, rejecting duplicate physical CSV headers before pandas
can mangle them, and normalizing serialized Boolean diagnostics without
silently accepting malformed flag values.
"""

from __future__ import annotations

import csv
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
_TRUE_BOOL_TEXT = frozenset({"true", "t", "yes", "y", "1", "1.0"})
_FALSE_BOOL_TEXT = frozenset(
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


def _opaque_sequence_identifier(value: str) -> object:
    """Preserve identifier text while retaining blank-field missingness."""

    return pd.NA if value == "" else value


def _physical_csv_columns(path: Path) -> list[str]:
    """Read the first non-blank physical CSV header without pandas mangling."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if row and not (len(row) == 1 and not row[0].strip()):
                return row
    return []


def _validate_unique_physical_columns(columns: list[str], *, path: Path) -> None:
    """Reject duplicate CSV headers before pandas silently renames them."""

    positions_by_name: dict[str, list[int]] = {}
    for position, name in enumerate(columns):
        positions_by_name.setdefault(name, []).append(position)
    collisions = {
        name: positions
        for name, positions in positions_by_name.items()
        if len(positions) > 1
    }
    if not collisions:
        return

    rendered = "; ".join(
        f"{name!r} at positions {positions}"
        for name, positions in collisions.items()
    )
    raise ValueError(f"{path} has duplicate physical CSV columns: {rendered}")


def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    """Load optional scorecard diagnostics while preserving opaque IDs."""

    if path is None:
        return None

    physical_columns = _physical_csv_columns(path)
    _validate_unique_physical_columns(physical_columns, path=path)
    identifier_columns = [
        name for name in _SEQUENCE_IDENTIFIER_DTYPES if name in physical_columns
    ]
    frame = pd.read_csv(
        path,
        converters={
            name: _opaque_sequence_identifier for name in identifier_columns
        },
    )
    for name in identifier_columns:
        frame[name] = frame[name].astype(_SEQUENCE_IDENTIFIER_DTYPES[name])
    return frame


def _bool_series(values: Any) -> pd.Series:
    """Normalize native and serialized Boolean diagnostics explicitly."""

    if values is None:
        return pd.Series(dtype=bool)
    input_name = getattr(values, "name", None)
    series = pd.Series(values, copy=False)
    if series.empty:
        return pd.Series(index=series.index, dtype=bool)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean").fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    truthy = (text.isin(_TRUE_BOOL_TEXT) | numeric.eq(1.0)).fillna(False)
    falsy = (
        series.isna() | text.isin(_FALSE_BOOL_TEXT) | numeric.eq(0.0)
    ).fillna(False)
    invalid = ~(truthy | falsy)
    if bool(invalid.any()):
        invalid_indices = invalid[invalid].index.tolist()
        invalid_values = series.loc[invalid_indices].tolist()
        label = str(input_name) if input_name is not None else "Boolean diagnostics"
        raise ValueError(
            f"{label} contains invalid Boolean values at rows "
            f"{invalid_indices}: {invalid_values}"
        )
    return truthy.astype(bool)


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
