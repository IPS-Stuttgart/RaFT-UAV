"""Compatibility wrapper validating Track 5 trajectory-smoother row keys.

The maintained implementation lives in the sibling ``track5_trajectory_smooth.py``
module. This package keeps the public import path while rejecting duplicate
``(sequence_id, time_s)`` rows before the local-linear fallback can associate a
row with a different estimate at the same timestamp.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_trajectory_smooth.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_trajectory_smooth_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 trajectory smoother from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_NORMALIZE = _IMPL._normalized_estimate_rows


def _normalized_estimate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize rows and reject ambiguous sequence/timestamp keys."""

    normalized = _ORIGINAL_NORMALIZE(rows)
    duplicate = normalized.duplicated(["sequence_id", "time_s"], keep=False)
    if duplicate.any():
        keys = (
            normalized.loc[duplicate, ["sequence_id", "time_s"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        preview = ", ".join(f"({sequence_id!r}, {time_s!r})" for sequence_id, time_s in keys)
        raise ValueError(f"Track 5 rows contain duplicate sequence/timestamp keys: {preview}")
    return normalized


_IMPL._normalized_estimate_rows = _normalized_estimate_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalized_estimate_rows"] = _normalized_estimate_rows

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
