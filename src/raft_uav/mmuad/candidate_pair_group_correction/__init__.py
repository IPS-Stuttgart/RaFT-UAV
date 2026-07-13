"""Compatibility wrapper preserving pair-group candidate input order.

The maintained implementation lives in the sibling
``candidate_pair_group_correction.py`` module. This package preserves the public
import path while restoring the original row order after schema normalization.
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
    """Normalize candidate rows without changing their relative input order."""

    rows = (
        candidates.rows.copy()
        if isinstance(candidates, _IMPL.CandidateFrame)
        else pd.DataFrame(candidates).copy()
    )
    marker = "__candidate_pair_input_order"
    while marker in rows.columns:
        marker = f"_{marker}"
    rows[marker] = np.arange(len(rows), dtype=int)
    marked_candidates = (
        _IMPL.CandidateFrame(rows)
        if isinstance(candidates, _IMPL.CandidateFrame)
        else rows
    )
    normalized = _ORIGINAL_CANDIDATE_ROWS(marked_candidates)
    if marker not in normalized.columns:
        return normalized
    return (
        normalized.sort_values(marker, kind="mergesort")
        .drop(columns=[marker])
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
