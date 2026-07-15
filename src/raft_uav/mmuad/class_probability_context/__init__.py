"""Compatibility wrapper requiring exact Track 5 predicted class IDs.

The maintained implementation lives in the sibling
``class_probability_context.py`` module. This package preserves the public
import path while preventing near-integer classifier labels from being rounded
into valid official class IDs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "class_probability_context.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._class_probability_context_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load class-probability context implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _predicted_class_labels(values: pd.Series) -> pd.Series:
    """Return canonical labels only for exactly integer-equivalent values."""

    raw = pd.Series(values)
    text = raw.where(raw.notna(), "").astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float)
    boolean_values = raw.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).to_numpy(dtype=bool)
    integer_like = (
        np.isfinite(numeric_array)
        & (numeric_array == np.rint(numeric_array))
        & ~boolean_values
    )
    if integer_like.any():
        positions = np.flatnonzero(integer_like)
        text.iloc[positions] = (
            np.rint(numeric_array[positions]).astype(int).astype(str)
        )
    return text


_IMPL._predicted_class_labels = _predicted_class_labels

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_predicted_class_labels"] = _predicted_class_labels

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
