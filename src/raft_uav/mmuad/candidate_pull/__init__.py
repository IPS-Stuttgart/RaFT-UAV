"""Compatibility package with stable candidate-pull normalization.

The maintained implementation lives in the sibling ``candidate_pull.py`` module.
This package preserves the public import path while canonicalizing official result
row indices, preserving result-row order during nearest-center alignment,
preserving opaque CSV sequence identifiers in the CLI, sanitizing non-finite
candidate ranking metadata, preventing finite score sums from overflowing during
candidate-center construction, and keeping candidate pulls temporally coherent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv

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

_ORIGINAL_MAIN = _IMPL.main
_ORIGINAL_NORMALIZE_OFFICIAL_RESULTS = _IMPL._normalize_official_results
_ORIGINAL_TOPK_CANDIDATE_CENTERS = _IMPL.topk_candidate_centers
_ORIGINAL_CANDIDATE_CENTERS_FOR_RESULTS = _IMPL.candidate_centers_for_results
_ORIGINAL_ALIGN_CANDIDATE_CENTERS = _IMPL.align_candidate_centers
_MAIN_LOCK = threading.RLock()
_CANDIDATE_SCORE_COLUMNS = (
    "ranker_score",
    "cluster_ranker_score",
    "candidate_ranker_score",
    "confidence",
    "score",
)
_SCORE_OUTPUT_COLUMNS = ("top_score", "top_score_margin")
_COORDINATE_COLUMNS = ("Timestamp", "x_m", "y_m", "z_m")


class _TextPreservingPandasProxy:
    """Delegate pandas operations while preserving opaque CSV identifiers."""

    def __init__(self, pandas_module: Any) -> None:
        self._pandas_module = pandas_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pandas_module, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if args or kwargs:
            return self._pandas_module.read_csv(path, *args, **kwargs)
        return read_estimate_csv(Path(path))


def main(argv: list[str] | None = None) -> int:
    """Run the candidate-pull CLI with text-preserving CSV parsing."""

    with _MAIN_LOCK:
        original_impl_pd = _IMPL.pd
        _IMPL.pd = _TextPreservingPandasProxy(original_impl_pd)
        try:
            return int(_ORIGINAL_MAIN(argv))
        finally:
            _IMPL.pd = original_impl_pd


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


def _nearest_candidate_frame(
    candidates: pd.DataFrame,
    *,
    sequence: object,
    target_time_s: object,
    tolerance_s: float,
) -> pd.DataFrame:
    """Return every hypothesis from the single nearest candidate timestamp."""

    if candidates.empty or not {"Sequence", "Timestamp"}.issubset(candidates.columns):
        return candidates.iloc[0:0].copy()
    sequence_rows = candidates.loc[
        candidates["Sequence"].astype(str) == str(sequence)
    ].copy()
    if sequence_rows.empty:
        return sequence_rows

    candidate_times = pd.to_numeric(sequence_rows["Timestamp"], errors="coerce")
    try:
        target_time = float(target_time_s)
    except (TypeError, ValueError, OverflowError):
        return sequence_rows.iloc[0:0].copy()
    finite = np.isfinite(candidate_times.to_numpy(dtype=float))
    if not np.isfinite(target_time) or not bool(finite.any()):
        return sequence_rows.iloc[0:0].copy()

    unique_times = np.unique(candidate_times.loc[finite].to_numpy(dtype=float))
    deltas = np.abs(unique_times - target_time)
    eligible = deltas <= float(tolerance_s)
    if not bool(eligible.any()):
        return sequence_rows.iloc[0:0].copy()
    eligible_times = unique_times[eligible]
    nearest_time = float(
        eligible_times[int(np.argmin(np.abs(eligible_times - target_time)))]
    )
    return sequence_rows.loc[candidate_times.eq(nearest_time)].copy()


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
    """Return row-wise centers from one complete nearest candidate frame.

    A tolerance window can contain several sensor timestamps. Ranking all rows in
    that window together treats detections from different times as simultaneous,
    so a high-score future or past candidate can replace every hypothesis from the
    actually nearest frame. Select one nearest timestamp per result row, use the
    earlier frame for exact ties, and retain every candidate from that timestamp.
    """

    rows = _sanitize_candidate_ranking_metadata(candidates)
    scaled, scales = _scale_sequence_scores(rows)
    result_rows = pd.DataFrame(results).copy()
    positions = np.asarray(current_xyz)
    parts: list[pd.DataFrame] = []
    for position, (row_index, result_row) in enumerate(result_rows.iterrows()):
        frame = _nearest_candidate_frame(
            scaled,
            sequence=result_row.get("Sequence"),
            target_time_s=result_row.get("Timestamp"),
            tolerance_s=float(time_tolerance_s),
        )
        if frame.empty:
            continue
        one_result = result_row.to_frame().T
        one_result.index = pd.RangeIndex(1)
        one_positions = positions[position : position + 1]
        centers = _ORIGINAL_CANDIDATE_CENTERS_FOR_RESULTS(
            frame,
            one_result,
            one_positions,
            top_k=top_k,
            time_tolerance_s=time_tolerance_s,
        )
        if centers.empty:
            continue
        centers = centers.copy()
        centers["row_index"] = row_index
        centers["Sequence"] = str(result_row.get("Sequence"))
        parts.append(centers)
    centers = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if centers.empty:
        return centers
    factors = np.array(
        [scales.get(str(sequence), 1.0) for sequence in centers["Sequence"]],
        dtype=float,
    )
    return _restore_score_scale(centers, factors)


def align_candidate_centers(
    results: pd.DataFrame,
    centers: pd.DataFrame,
    *,
    time_tolerance_s: float,
) -> pd.DataFrame:
    """Align centers while preserving the caller's result-row order.

    The legacy implementation sorts rows inside each sequence and concatenates
    sequences in group-key order. Its returned frame therefore no longer matches
    the positional order of ``results`` when sequences are interleaved or
    timestamps are unsorted. The high-level candidate-pull workflow assigns
    uncertainty columns from that frame positionally, which can attach another
    row's diagnostics. Carrying a temporary global order column through the
    merge restores the original row order after all per-sequence alignments.
    """

    ordered_results = results.copy()
    order_column = "__candidate_pull_result_order__"
    occupied = set(ordered_results.columns) | set(centers.columns)
    while order_column in occupied:
        order_column += "_"
    ordered_results[order_column] = np.arange(len(ordered_results), dtype=np.int64)
    aligned = _ORIGINAL_ALIGN_CANDIDATE_CENTERS(
        ordered_results,
        centers,
        time_tolerance_s=time_tolerance_s,
    )
    if order_column not in aligned.columns:
        return aligned
    return (
        aligned.sort_values(order_column, kind="mergesort")
        .drop(columns=[order_column])
        .reset_index(drop=True)
    )


_IMPL.main = main
_IMPL._normalize_official_results = _normalize_official_results
_IMPL.topk_candidate_centers = topk_candidate_centers
_IMPL.candidate_centers_for_results = candidate_centers_for_results
_IMPL.align_candidate_centers = align_candidate_centers

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["main"] = main
globals()["_normalize_official_results"] = _normalize_official_results
globals()["_nearest_candidate_frame"] = _nearest_candidate_frame
globals()["topk_candidate_centers"] = topk_candidate_centers
globals()["candidate_centers_for_results"] = candidate_centers_for_results
globals()["align_candidate_centers"] = align_candidate_centers

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
