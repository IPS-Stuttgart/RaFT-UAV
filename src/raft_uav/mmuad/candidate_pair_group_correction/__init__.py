"""Compatibility wrapper preserving pair-group candidate row order.

The maintained implementation lives in the sibling
``candidate_pair_group_correction.py`` module. Canonical candidate
normalization sorts rows by sequence, timestamp, and source; pair-group
preparation, however, is expected to preserve the caller's candidate order.
This package restores that order after the established normalization and
filtering logic has run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_pair_group_correction.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_pair_group_correction_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load pair-group correction implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_CANDIDATE_ROWS = _IMPL._candidate_rows


def _candidate_rows(candidates):
    """Normalize/filter candidates without changing surviving row order."""

    raw = (
        candidates.rows.copy()
        if isinstance(candidates, _IMPL.CandidateFrame)
        else pd.DataFrame(candidates).copy()
    )
    if raw.empty:
        return _ORIGINAL_CANDIDATE_ROWS(candidates)

    order_column = "__candidate_pair_original_row_order"
    while order_column in raw.columns:
        order_column = f"_{order_column}"
    raw[order_column] = np.arange(len(raw), dtype=np.int64)

    normalized = _ORIGINAL_CANDIDATE_ROWS(raw)
    if order_column not in normalized.columns:
        return normalized
    return (
        normalized.sort_values(order_column, kind="stable")
        .drop(columns=[order_column])
        .reset_index(drop=True)
    )


_IMPL._candidate_rows = _candidate_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_candidate_rows"] = _candidate_rows
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
