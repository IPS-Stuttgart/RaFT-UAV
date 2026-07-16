"""Compatibility fix for Track 5 classification relabel validation.

The maintained implementation lives in the sibling
``track5_classification_relabel.py`` module. This package preserves the public
import path while requiring class labels to be exactly integer-equivalent before
relabeling.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_classification_relabel.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_classification_relabel_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 classification relabel implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

VALID_CLASS_IDS = _IMPL.VALID_CLASS_IDS


def _validate_class_series(values: pd.Series, *, name: str) -> None:
    """Require finite labels exactly equal to official integer class IDs."""

    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric_values = numeric.to_numpy(float)
    if numeric.isna().any() or not np.isfinite(numeric_values).all():
        raise ValueError(f"{name} contains non-finite class labels")

    rounded_values = np.rint(numeric_values)
    if not np.equal(numeric_values, rounded_values).all():
        raise ValueError(f"{name} contains non-integer class labels")

    integer_values = pd.Series(
        rounded_values.astype(int),
        index=numeric.index,
    )
    bad = sorted(
        set(
            integer_values.loc[~integer_values.isin(VALID_CLASS_IDS)]
            .astype(int)
            .tolist()
        )
    )
    if bad:
        allowed = ", ".join(str(class_id) for class_id in VALID_CLASS_IDS)
        raise ValueError(f"{name} contains class labels outside {{{allowed}}}: {bad}")


_IMPL._validate_class_series = _validate_class_series

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_class_series"] = _validate_class_series

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
