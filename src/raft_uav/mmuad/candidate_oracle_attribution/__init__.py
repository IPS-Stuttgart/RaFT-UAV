"""Compatibility validation and deterministic ordering for candidate-oracle attribution.

The maintained implementation lives in the sibling
``candidate_oracle_attribution.py`` module. This package preserves the public
import path while rejecting malformed truth-matching time gates and top-K values
before they can silently widen, empty, or change the diagnostic. Equal-score
candidates are ranked with explicit stable tie-break keys so CSV row order cannot
change top-K oracle attribution. Duplicate truth timestamps retain the final
original row so attribution follows the repository-wide authoritative trajectory
convention. The CLI also reads truth tables through the shared text-preserving
MMUAD CSV reader so opaque sequence identifiers remain aligned with candidate
inputs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_attribution.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_attribution_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load candidate-oracle attribution implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_MAIN = _IMPL.main
_MAIN_LOCK = threading.RLock()
_RANK_TEXT_COLUMNS = (
    ("source", "unknown"),
    ("track_id", ""),
    ("candidate_branch", "candidate"),
    ("class_name", ""),
)


class _TextPreservingPandasProxy:
    """Delegate pandas operations while preserving opaque truth identifiers."""

    def __init__(self, pandas_module: Any) -> None:
        self._pandas_module = pandas_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pandas_module, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if args or kwargs:
            return self._pandas_module.read_csv(path, *args, **kwargs)
        return read_estimate_csv(Path(path))


def _nonnegative_finite_scalar(value: object, *, name: str) -> float:
    """Return a validated non-negative finite scalar control."""

    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be a non-negative finite scalar")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative finite scalar") from exc
    if array.ndim != 0 or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a non-negative finite scalar")
    try:
        normalized = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative finite scalar") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a non-negative finite scalar")
    return normalized


def _normalize_top_k_values(values: Sequence[object]) -> tuple[int, ...]:
    """Return sorted unique positive integer top-K values without truncation."""

    message = "top_k_values must contain only positive integers"
    if isinstance(values, (str, bytes)):
        raise ValueError(message)
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(message) from exc

    normalized: list[int] = []
    for value in raw_values:
        try:
            number = _nonnegative_finite_scalar(value, name="top_k_values")
        except ValueError as exc:
            raise ValueError(message) from exc
        if number <= 0.0 or not number.is_integer():
            raise ValueError(message)
        normalized.append(int(number))
    return tuple(sorted(set(normalized)))


def _normalize_truth_trajectory(truth: pd.DataFrame) -> pd.DataFrame:
    """Normalize truth and retain the final original row at duplicate times."""

    raw_truth = pd.DataFrame(truth).copy()
    order_column = "_candidate_oracle_attribution_input_order"
    while order_column in raw_truth.columns:
        order_column = f"_{order_column}"
    raw_truth[order_column] = np.arange(len(raw_truth), dtype=np.int64)
    truth_rows = _IMPL.normalize_truth_columns(raw_truth)
    if truth_rows.empty:
        return truth_rows.drop(columns=[order_column], errors="ignore")

    return (
        truth_rows.sort_values(
            ["sequence_id", "time_s", order_column],
            kind="mergesort",
        )
        .drop_duplicates(["sequence_id", "time_s"], keep="last")
        .sort_values(["sequence_id", "time_s"], kind="mergesort")
        .drop(columns=[order_column])
        .reset_index(drop=True)
    )


def _stable_text_column(rows: pd.DataFrame, column: str, *, default: str) -> pd.Series:
    """Return one comparable deterministic text key for candidate ranking."""

    if column not in rows.columns:
        return pd.Series(default, index=rows.index, dtype="string")
    values = rows[column].where(rows[column].notna(), default)
    return values.astype(str).str.strip()


def _deterministic_candidate_ranking(group: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by score and explicit stable tie-break keys."""

    ranked = group.copy()
    helper_columns: list[str] = []
    for column, default in _RANK_TEXT_COLUMNS:
        helper = f"_oracle_rank_{column}"
        ranked[helper] = _stable_text_column(ranked, column, default=default)
        helper_columns.append(helper)

    sort_columns = [
        "candidate_oracle_score",
        *helper_columns,
        "x_m",
        "y_m",
        "z_m",
    ]
    ascending = [False, *([True] * (len(sort_columns) - 1))]
    return (
        ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        .drop(columns=helper_columns)
        .reset_index(drop=True)
    )


def build_candidate_oracle_attribution_tables(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    top_k_values: Sequence[int] = _IMPL._DEFAULT_TOP_K,
    score_column: str = "candidate_reservoir_score",
    fallback_score_column: str = "ranker_score",
    max_truth_time_delta_s: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return oracle-attribution tables with validated deterministic ranking."""

    max_delta = _nonnegative_finite_scalar(
        max_truth_time_delta_s,
        name="max_truth_time_delta_s",
    )
    top_k = _normalize_top_k_values(top_k_values)
    rows = _IMPL.normalize_candidate_columns(pd.DataFrame(candidates).copy())
    truth_rows = _normalize_truth_trajectory(truth)
    if rows.empty or truth_rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    rows = rows.copy()
    if "source" not in rows.columns:
        rows["source"] = "unknown"
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows["source"].fillna("candidate").astype(str)
    if "track_id" not in rows.columns:
        rows["track_id"] = np.arange(len(rows), dtype=int).astype(str)
    rows["candidate_oracle_score"] = _IMPL._score_column(
        rows,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
    )
    truth_by_sequence = {
        str(sequence_id): group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in truth_rows.groupby("sequence_id", sort=True)
    }

    frame_records: list[dict[str, Any]] = []
    for (sequence_id, time_s), group in rows.groupby(
        ["sequence_id", "time_s"],
        sort=True,
    ):
        seq_truth = truth_by_sequence.get(str(sequence_id))
        if seq_truth is None or seq_truth.empty:
            continue
        truth_t = seq_truth["time_s"].to_numpy(float)
        nearest_idx = int(np.argmin(np.abs(truth_t - float(time_s))))
        truth_dt = float(time_s) - float(truth_t[nearest_idx])
        if abs(truth_dt) > max_delta:
            continue
        truth_xyz = seq_truth.iloc[nearest_idx][["x_m", "y_m", "z_m"]].to_numpy(float)
        ranked = _deterministic_candidate_ranking(group)
        candidate_xyz = ranked[["x_m", "y_m", "z_m"]].to_numpy(float)
        distances = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
        best_pos = int(np.argmin(distances))
        best_row = ranked.iloc[best_pos]
        record: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "time_s": float(time_s),
            "truth_time_delta_s": truth_dt,
            "candidate_count": int(len(ranked)),
            "oracle_all_3d_m": float(distances[best_pos]),
            "oracle_all_rank": int(best_pos + 1),
            "oracle_all_rank_fraction": float((best_pos + 1) / max(len(ranked), 1)),
            "oracle_all_candidate_score": float(best_row["candidate_oracle_score"]),
            "oracle_all_candidate_source": str(best_row.get("source", "unknown")),
            "oracle_all_candidate_branch": str(
                best_row.get("candidate_branch", "candidate")
            ),
            "oracle_all_candidate_track_id": str(best_row.get("track_id", "")),
        }
        for top_k_value in top_k:
            bounded_k = min(int(top_k_value), len(distances))
            top_distances = distances[:bounded_k]
            top_best_pos = int(np.argmin(top_distances))
            top_row = ranked.iloc[top_best_pos]
            record[f"oracle_top{top_k_value}_3d_m"] = float(
                top_distances[top_best_pos]
            )
            record[f"oracle_in_top{top_k_value}"] = bool(best_pos < bounded_k)
            record[f"oracle_top{top_k_value}_candidate_source"] = str(
                top_row.get("source", "unknown")
            )
            record[f"oracle_top{top_k_value}_candidate_branch"] = str(
                top_row.get("candidate_branch", "candidate")
            )
        frame_records.append(record)

    frame_rows = pd.DataFrame.from_records(frame_records)
    if frame_rows.empty:
        empty = pd.DataFrame()
        return frame_rows, empty, empty, empty
    pooled = pd.DataFrame.from_records(
        [_IMPL._pooled_summary(frame_rows, top_k_values=top_k)]
    )
    branch_summary = _IMPL._group_summary(
        frame_rows,
        group_column="oracle_all_candidate_branch",
        label_column="candidate_branch",
    )
    source_summary = _IMPL._group_summary(
        frame_rows,
        group_column="oracle_all_candidate_source",
        label_column="source",
    )
    return frame_rows, pooled, branch_summary, source_summary


def main(argv: list[str] | None = None) -> int:
    """Run the CLI with text-preserving truth CSV parsing."""

    with _MAIN_LOCK:
        original_impl_pd = _IMPL.pd
        _IMPL.pd = _TextPreservingPandasProxy(original_impl_pd)
        try:
            return int(_ORIGINAL_MAIN(argv))
        finally:
            _IMPL.pd = original_impl_pd


_IMPL.build_candidate_oracle_attribution_tables = (
    build_candidate_oracle_attribution_tables
)
_IMPL._normalize_top_k_values = _normalize_top_k_values
_IMPL._deterministic_candidate_ranking = _deterministic_candidate_ranking
_IMPL.main = main

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_top_k_values"] = _normalize_top_k_values
globals()["_deterministic_candidate_ranking"] = _deterministic_candidate_ranking
globals()["build_candidate_oracle_attribution_tables"] = (
    build_candidate_oracle_attribution_tables
)
globals()["main"] = main

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
