"""Compatibility fixes for pair-group score preparation.

The maintained implementation lives in the sibling
``candidate_pair_group_correction.py`` module. This package preserves the public
import path while restoring input order after schema normalization and resolving
configured score fallbacks independently for every candidate row.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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


def _candidate_scores_with_rowwise_fallback(
    rows: pd.DataFrame,
    config: Any,
) -> pd.Series:
    """Resolve configured score columns independently for each candidate row."""

    resolved = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in (config.score_column, *config.fallback_score_columns):
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        values = values.where(np.isfinite(values))
        resolved = resolved.where(resolved.notna(), values)

    finite = resolved.dropna()
    fill_value = float(finite.min()) if not finite.empty else 1.0
    return resolved.fillna(fill_value).astype(float)


_IMPL._candidate_rows = _candidate_rows
_IMPL._candidate_scores = _candidate_scores_with_rowwise_fallback

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_candidate_rows"] = _candidate_rows
globals()["_candidate_scores"] = _candidate_scores_with_rowwise_fallback

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
