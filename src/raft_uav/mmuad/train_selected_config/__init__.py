"""Compatibility wrapper for strict train-selected configuration handling.

The maintained implementation lives in the sibling ``train_selected_config.py``
module. This package preserves the public import path while making alias
selection skip missing values and rejecting malformed numeric controls before
Python or NumPy can silently coerce them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "train_selected_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._train_selected_config_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load train-selected config implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _first_present(row: pd.Series, columns: tuple[str, ...]) -> Any:
    """Return the first present, non-missing alias value from ``row``."""

    for column in columns:
        if column not in row.index:
            continue
        value = row[column]
        if not _IMPL._is_nan(value):
            return value
    return None


def _unwrap_numeric_scalar(value: Any, *, message: str) -> Any:
    """Recursively unwrap zero-dimensional arrays without scalar coercion."""

    seen_array_ids: set[int] = set()
    while isinstance(value, np.ndarray):
        array_id = id(value)
        if array_id in seen_array_ids:
            raise ValueError(message)
        seen_array_ids.add(array_id)

        if np.ma.isMaskedArray(value):
            if bool(np.ma.getmaskarray(value).any()):
                raise ValueError(message)
            value = np.ma.getdata(value)
            continue
        if value.ndim != 0:
            raise ValueError(message)
        try:
            value = value.item()
        except ValueError as exc:
            raise ValueError(message) from exc
    return value


def _float(value: Any) -> float:
    """Return a finite scalar float without lossy implicit coercion."""

    message = f"expected finite float, got {value!r}"
    if np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = _unwrap_numeric_scalar(value, message=message)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if isinstance(scalar, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(message)
    if np.ma.is_masked(scalar):
        raise ValueError(message)
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if not np.isfinite(number):
        raise ValueError(message)
    return number


_IMPL._first_present = _first_present
_IMPL._float = _float

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_first_present"] = _first_present
globals()["_unwrap_numeric_scalar"] = _unwrap_numeric_scalar
globals()["_float"] = _float

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
