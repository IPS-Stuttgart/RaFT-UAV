"""Compatibility fixes for pair-group score preparation.

The maintained implementation lives in the sibling
``candidate_pair_group_correction.py`` module. This package preserves the public
import path while making pre-inference score preparation resolve fallbacks row
by row, treating tied ranks symmetrically, and preserving candidate input order.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame

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


def _candidate_scores_with_rowwise_fallback(rows: pd.DataFrame, config: Any) -> pd.Series:
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


def _normalize_scores_with_average_ties(values: np.ndarray, mode: str) -> np.ndarray:
    """Normalize scores without using candidate row order to break rank ties."""

    score = np.asarray(values, dtype=float)
    finite = np.isfinite(score)
    if not finite.any():
        return np.zeros_like(score, dtype=float)
    floor = float(np.min(score[finite]))
    score = np.where(finite, score, floor)
    if mode == "none":
        return score
    if mode == "rank":
        if len(score) <= 1:
            return np.full(len(score), 0.5, dtype=float)
        ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
        return (ranks - 1.0) / float(len(score) - 1)
    minimum = float(np.min(score))
    maximum = float(np.max(score))
    if maximum <= minimum:
        return np.full(len(score), 0.5, dtype=float)
    return (score - minimum) / (maximum - minimum)


_ORIGINAL_CANDIDATE_ROWS = _IMPL._candidate_rows


def _candidate_rows_preserving_input_order(
    candidates: CandidateFrame | pd.DataFrame,
) -> pd.DataFrame:
    """Normalize candidate rows without retaining schema-level source sorting."""

    rows = (
        candidates.rows.copy()
        if isinstance(candidates, CandidateFrame)
        else pd.DataFrame(candidates).copy()
    )
    if rows.empty:
        return _ORIGINAL_CANDIDATE_ROWS(rows)

    order_column = "__candidate_pair_group_input_order"
    while order_column in rows.columns:
        order_column = f"_{order_column}"
    rows[order_column] = np.arange(len(rows), dtype=int)

    normalized = _ORIGINAL_CANDIDATE_ROWS(rows)
    if order_column not in normalized.columns:
        return normalized
    return (
        normalized.sort_values(order_column, kind="stable")
        .drop(columns=[order_column])
        .reset_index(drop=True)
    )


_IMPL._candidate_scores = _candidate_scores_with_rowwise_fallback
_IMPL._normalize_scores = _normalize_scores_with_average_ties
_IMPL._candidate_rows = _candidate_rows_preserving_input_order

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
