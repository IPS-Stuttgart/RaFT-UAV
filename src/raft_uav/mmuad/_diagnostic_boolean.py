"""Strict parsing helpers for persisted Boolean diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

_INVALID_BOOLEAN = object()
_TRUE_BOOLEAN_TOKENS = frozenset({"true", "t", "yes", "y", "on"})
_FALSE_BOOLEAN_TOKENS = frozenset({"false", "f", "no", "n", "off"})
_MISSING_BOOLEAN_TOKENS = frozenset({"", "nan", "none", "null", "<na>", "nat"})


def _boolean_number(value: float) -> bool | object:
    if np.isnan(value):
        return False
    if not np.isfinite(value):
        return _INVALID_BOOLEAN
    if value == 0.0:
        return False
    if value == 1.0:
        return True
    return _INVALID_BOOLEAN


def _boolean_cell(value: object) -> bool | object:
    if value is None or value is pd.NA or np.ma.is_masked(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in _TRUE_BOOLEAN_TOKENS:
            return True
        if text in _FALSE_BOOLEAN_TOKENS or text in _MISSING_BOOLEAN_TOKENS:
            return False
        try:
            return _boolean_number(float(text))
        except (TypeError, ValueError, OverflowError):
            return _INVALID_BOOLEAN

    try:
        array = np.asanyarray(value)
    except (TypeError, ValueError):
        return _INVALID_BOOLEAN
    if array.ndim != 0 or np.iscomplexobj(array):
        return _INVALID_BOOLEAN
    if np.ma.isMaskedArray(array) and bool(np.ma.getmaskarray(array).any()):
        return False

    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    try:
        if bool(pd.isna(scalar)):
            return False
    except (TypeError, ValueError):
        return _INVALID_BOOLEAN
    try:
        return _boolean_number(float(scalar))
    except (TypeError, ValueError, OverflowError):
        return _INVALID_BOOLEAN


def normalize_persisted_boolean_column(values: pd.Series, *, column: str) -> pd.Series:
    """Return strict Boolean values while preserving the input row index."""

    series = pd.Series(values, copy=False)
    normalized: list[bool] = []
    invalid: list[tuple[object, object]] = []
    for index, value in series.items():
        parsed = _boolean_cell(value)
        if parsed is _INVALID_BOOLEAN:
            invalid.append((index, value))
            normalized.append(False)
        else:
            normalized.append(bool(parsed))

    if invalid:
        preview = ", ".join(
            f"index {index!r}: {value!r}" for index, value in invalid[:5]
        )
        suffix = "" if len(invalid) <= 5 else f"; plus {len(invalid) - 5} more"
        raise ValueError(f"{column} contains invalid Boolean values: {preview}{suffix}")
    return pd.Series(
        normalized,
        index=series.index,
        name=series.name,
        dtype=bool,
    )
