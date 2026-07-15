"""Compatibility package with stable candidate-pull normalization.

The maintained implementation lives in the sibling ``candidate_pull.py`` module.
This package preserves the public import path while canonicalizing official result
row indices and sanitizing non-finite candidate ranking metadata before positional
candidate-center alignment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_pull.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_pull_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate-pull implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE_OFFICIAL_RESULTS = _IMPL._normalize_official_results
_ORIGINAL_TOPK_CANDIDATE_CENTERS = _IMPL.topk_candidate_centers
_ORIGINAL_CANDIDATE_CENTERS_FOR_RESULTS = _IMPL.candidate_centers_for_results
_CANDIDATE_SCORE_COLUMNS = (
    "ranker_score",
    "cluster_ranker_score",
    "candidate_ranker_score",
    "confidence",
    "score",
)


def _normalize_official_results(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return official rows with indices matching positional coordinate arrays.

    The legacy implementation derives ``xyz`` positionally but retains the input
    DataFrame's index labels. Downstream candidate-center logic uses those labels
    to index ``xyz``, so filtered or concatenated frames can address the wrong row
    or raise ``IndexError``. Resetting the internal row index keeps both views in
    the same coordinate system without changing row order or output values.
    """

    rows, xyz = _ORIGINAL_NORMALIZE_OFFICIAL_RESULTS(results)
    return rows.reset_index(drop=True), xyz


def _sanitize_candidate_ranking_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    """Treat non-finite candidate ranking metadata as missing values.

    The maintained implementation already converts missing ranking metadata to
    zero, but positive infinity survives ``to_numeric`` and can outrank every
    finite candidate. Infinite scores also produce ``inf / inf`` during weighted
    center construction. Replacing all non-finite ranking values with ``NaN``
    preserves the existing missing-value fallback without discarding valid
    candidate coordinates.
    """

    rows = pd.DataFrame(candidates).copy()
    _IMPL._rename_candidate_columns(rows)
    ranking_columns = (*_CANDIDATE_SCORE_COLUMNS, "cluster_point_count")
    for column in ranking_columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        rows[column] = values.where(np.isfinite(values), np.nan)
    return rows


def topk_candidate_centers(
    candidates: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return candidate centers without non-finite ranking contamination."""

    return _ORIGINAL_TOPK_CANDIDATE_CENTERS(
        _sanitize_candidate_ranking_metadata(candidates),
        top_k=top_k,
    )


def candidate_centers_for_results(
    candidates: pd.DataFrame,
    results: pd.DataFrame,
    current_xyz: np.ndarray,
    *,
    top_k: int = 5,
    time_tolerance_s: float = 0.5,
) -> pd.DataFrame:
    """Return row-wise centers after sanitizing candidate ranking metadata."""

    return _ORIGINAL_CANDIDATE_CENTERS_FOR_RESULTS(
        _sanitize_candidate_ranking_metadata(candidates),
        results,
        current_xyz,
        top_k=top_k,
        time_tolerance_s=time_tolerance_s,
    )


_IMPL._normalize_official_results = _normalize_official_results
_IMPL.topk_candidate_centers = topk_candidate_centers
_IMPL.candidate_centers_for_results = candidate_centers_for_results

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_official_results"] = _normalize_official_results
globals()["topk_candidate_centers"] = topk_candidate_centers
globals()["candidate_centers_for_results"] = candidate_centers_for_results

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
