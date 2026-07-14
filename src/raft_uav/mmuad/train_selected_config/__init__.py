"""Compatibility wrapper for missing-aware train-summary aliases.

The maintained implementation lives in the sibling ``train_selected_config.py``
module. This package preserves the public import path while making alias
selection skip missing values, so a blank canonical column does not hide a
populated legacy alias in the same selected summary row.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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


_IMPL._first_present = _first_present

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
