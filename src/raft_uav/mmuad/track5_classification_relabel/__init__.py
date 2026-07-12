"""Compatibility fixes for Track 5 classification relabelling.

The maintained implementation lives in the sibling
``track5_classification_relabel.py`` module. This package preserves the public
import path while rejecting non-finite nearest-time tolerances before they can
silently disable the relabelling gate through IEEE comparison semantics.
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
    raise ImportError(f"cannot load Track 5 classification relabeller from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NEAREST_TIME_RELABEL_MERGE = _IMPL._nearest_time_relabel_merge


def _nearest_time_relabel_merge(
    pose: pd.DataFrame,
    source: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None,
) -> pd.DataFrame:
    """Reject NaN and infinite nearest-time gates before matching rows."""

    if max_nearest_time_delta_s is not None:
        tolerance = float(max_nearest_time_delta_s)
        if not np.isfinite(tolerance):
            raise ValueError("max_nearest_time_delta_s must be finite")
        max_nearest_time_delta_s = tolerance
    return _ORIGINAL_NEAREST_TIME_RELABEL_MERGE(
        pose,
        source,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )


_IMPL._nearest_time_relabel_merge = _nearest_time_relabel_merge

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_nearest_time_relabel_merge"] = _nearest_time_relabel_merge
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
