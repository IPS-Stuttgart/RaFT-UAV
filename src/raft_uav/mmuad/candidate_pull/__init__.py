"""Compatibility package with stable candidate-pull normalization.

The maintained implementation lives in the sibling ``candidate_pull.py`` module.
This package preserves the public import path while canonicalizing official result
row indices, sanitizing non-finite candidate ranking metadata, and preventing
finite score sums from overflowing during candidate-center construction.
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
_SCORE_OUTPUT_COLUMNS = ("top_score", "top_score_margin")
_COORDINATE_COLUMNS = ("Timestamp", "x_m", "y_m", "z_m")


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


def _finite_candidate_mask(rows: pd.DataFrame) -> pd.Series:
    """Return rows whose time and candidate coordinates are finite."""

    if not set(_COORDINATE_COLUMNS).issubset(rows.columns):
        return pd.Series(False, index=rows.index, dtype=bool)
    values = rows[list(_COORDINATE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return pd.Series(
        np.isfinite(values.to_numpy(dtype=float)).all(axis=1),
        index=rows.index,
        dtype=bool,
    )


def _overflow_safe_scale(values: np.ndarray, *, count: int) -> float:
    """Return the smallest scale that keeps a nonnegative sum finite."""

    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive.size == 0:
        return 1.0
    maximum = float(np.max(finite_positive))
    safe_term_limit = np.finfo(float).max / (2.0 * float(max(int(count), 1)))
    return maximum / safe_term_limit if maximum > safe_term_limit else 1.0


def _scale_topk_scores(
    rows: pd.DataFrame,
    score_column: str,
) -> tuple[pd.DataFrame, dict[tuple[str, float], float]]:
    """Scale scores per frame enough to prevent finite sum overflow."""

    scaled = rows.reset_index(drop=True).copy()
    if not {"Sequence", "Timestamp", score_column}.issubset(scaled.columns):
        return scaled, {}
    valid = _finite_candidate_mask(scaled)
    scales: dict[tuple[str, float], float] = {}
    grouped = scaled.loc[valid].groupby(["Sequence", "Timestamp"], sort=False)
    for (sequence, timestamp), group in grouped:
        values = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
        scale = _overflow_safe_scale(values, count=len(group))
        scales[(str(sequence), float(timestamp))] = scale
        if scale > 1.0:
            scaled.loc[group.index, score_column] = values / scale
    return scaled, scales


def _scale_sequence_scores(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Scale ranker scores per sequence enough for every windowed sum."""

    scaled = rows.reset_index(drop=True).copy()
    if not {"Sequence", "ranker_score"}.issubset(scaled.columns):
        return scaled, {}
    valid = _finite_candidate_mask(scaled)
    scales: dict[str, float] = {}
    for sequence, group in scaled.loc[valid].groupby("Sequence", sort=False):
        values = pd.to_numeric(group["ranker_score"], errors="coerce").to_numpy(dtype=float)
        scale = _overflow_safe_scale(values, count=len(group))
        scales[str(sequence)] = scale
        if scale > 1.0:
            scaled.loc[group.index, "ranker_score"] = values / scale
    return scaled, scales


def _restore_score_scale(frame: pd.DataFrame, factors: np.ndarray) -> pd.DataFrame:
    """Restore diagnostic score units after overflow-safe normalization."""

    restored = frame.copy()
    for column in _SCORE_OUTPUT_COLUMNS:
        if column not in restored.columns:
            continue
        values = pd.to_numeric(restored[column], errors="coerce").to_numpy(dtype=float)
        restored[column] = values * factors
    return restored


def topk_candidate_centers(
    candidates: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return candidate centers without invalid or overflowing score weights."""

    rows = _sanitize_candidate_ranking_metadata(candidates)
    score_column = _IMPL._first_existing_column(rows, _CANDIDATE_SCORE_COLUMNS)
    if score_column is None:
        return _ORIGINAL_TOPK_CANDIDATE_CENTERS(rows, top_k=top_k)
    scaled, scales = _scale_topk_scores(rows, score_column)
    centers = _ORIGINAL_TOPK_CANDIDATE_CENTERS(scaled, top_k=top_k)
    if centers.empty:
        return centers
    factors = np.array(
        [
            scales.get((str(row["Sequence"]), float(row["candidate_time_s"])), 1.0)
            for _, row in centers.iterrows()
        ],
        dtype=float,
    )
    return _restore_score_scale(centers, factors)


def candidate_centers_for_results(
    candidates: pd.DataFrame,
    results: pd.DataFrame,
    current_xyz: np.ndarray,
    *,
    top_k: int = 5,
    time_tolerance_s: float = 0.5,
) -> pd.DataFrame:
    """Return row-wise centers with stable finite score normalization."""

    rows = _sanitize_candidate_ranking_metadata(candidates)
    scaled, scales = _scale_sequence_scores(rows)
    centers = _ORIGINAL_CANDIDATE_CENTERS_FOR_RESULTS(
        scaled,
        results,
        current_xyz,
        top_k=top_k,
        time_tolerance_s=time_tolerance_s,
    )
    if centers.empty:
        return centers
    factors = np.array(
        [scales.get(str(sequence), 1.0) for sequence in centers["Sequence"]],
        dtype=float,
    )
    return _restore_score_scale(centers, factors)


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
