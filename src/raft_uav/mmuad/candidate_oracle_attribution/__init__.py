"""Compatibility validation and deterministic ordering for candidate-oracle attribution.

The maintained implementation lives in the sibling
``candidate_oracle_attribution.py`` module. This package preserves the public
import path while rejecting malformed truth-matching time gates and top-K values
before they can silently widen, empty, or change the diagnostic. Equal-score
candidates are ranked with explicit stable tie-break keys so CSV row order cannot
change top-K oracle attribution. Duplicate truth timestamps retain the final
original row within each physical flight so attribution follows the repository-
wide authoritative trajectory convention. Candidate/truth matching is scoped by
joint ``(sequence_id, flight_id)`` whenever flight metadata is available. The CLI
also reads truth tables through the shared text-preserving MMUAD CSV reader so
opaque sequence identifiers remain aligned with candidate inputs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

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
_MISSING_SCOPE_TEXT = frozenset({"", "nan", "none", "null", "<na>", "nat"})


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


def _canonical_scope_value(value: object, *, role: str) -> str | None:
    """Return one normalized flight identifier or ``None`` when it is missing."""

    if not is_scalar(value):
        raise ValueError(
            f"candidate-oracle attribution requires scalar flight_id values on {role} rows"
        )
    if value is None or bool(pd.isna(value)):
        return None
    text = str(value).strip()
    return None if text.casefold() in _MISSING_SCOPE_TEXT else text


def _normalized_flight_ids(frame: pd.DataFrame, *, role: str) -> pd.Series | None:
    """Return complete meaningful flight IDs, or ``None`` when metadata is absent."""

    if "flight_id" not in frame.columns:
        return None
    try:
        values = frame["flight_id"].map(
            lambda value: _canonical_scope_value(value, role=role)
        ).astype(object)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"candidate-oracle attribution requires scalar flight_id values on {role} rows"
        ) from exc
    if not bool(values.notna().any()):
        return None
    if bool(values.isna().any()):
        raise ValueError(
            "candidate-oracle attribution requires flight_id on every "
            f"{role} row when flight metadata is provided"
        )
    return values


def _normalize_truth_trajectory(truth: pd.DataFrame) -> pd.DataFrame:
    """Normalize truth and retain the final row per physical scope and timestamp."""

    raw_truth = pd.DataFrame(truth).copy()
    order_column = "_candidate_oracle_attribution_input_order"
    while order_column in raw_truth.columns:
        order_column = f"_{order_column}"
    raw_truth[order_column] = np.arange(len(raw_truth), dtype=np.int64)
    truth_rows = _IMPL.normalize_truth_columns(raw_truth)
    if truth_rows.empty:
        return truth_rows.drop(columns=[order_column], errors="ignore")

    scope_columns = ["sequence_id"]
    flight_ids = _normalized_flight_ids(truth_rows, role="truth")
    if flight_ids is not None:
        truth_rows = truth_rows.copy()
        truth_rows["flight_id"] = flight_ids
        scope_columns.append("flight_id")
    dedup_columns = [*scope_columns, "time_s"]
    return (
        truth_rows.sort_values(
            [*dedup_columns, order_column],
            kind="mergesort",
        )
        .drop_duplicates(dedup_columns, keep="last")
        .sort_values(dedup_columns, kind="mergesort")
        .drop(columns=[order_column])
        .reset_index(drop=True)
    )


def _single_flight_by_sequence(
    frame: pd.DataFrame,
    flight_ids: pd.Series,
    *,
    role: str,
) -> dict[str, str]:
    """Return a sequence-to-flight map when one-sided metadata is unambiguous."""

    scoped = pd.DataFrame(
        {
            "sequence_id": frame["sequence_id"].astype(str),
            "flight_id": flight_ids.astype(str),
        },
        index=frame.index,
    )
    mapping: dict[str, str] = {}
    for sequence_id, group in scoped.groupby("sequence_id", sort=False):
        flights = tuple(dict.fromkeys(group["flight_id"].tolist()))
        if len(flights) > 1:
            raise ValueError(
                "candidate-oracle attribution cannot align one-sided flight_id "
                f"metadata: {role} sequence {sequence_id!r} contains multiple "
                f"physical flights {list(flights)!r}"
            )
        mapping[str(sequence_id)] = str(flights[0])
    return mapping


def _align_physical_scope(
    rows: pd.DataFrame,
    truth_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Align candidates and truth to the same physical-flight scope."""

    candidate_rows = rows.copy()
    aligned_truth = truth_rows.copy()
    candidate_flights = _normalized_flight_ids(candidate_rows, role="candidate")
    truth_flights = _normalized_flight_ids(aligned_truth, role="truth")

    if candidate_flights is not None:
        candidate_rows["flight_id"] = candidate_flights
    if truth_flights is not None:
        aligned_truth["flight_id"] = truth_flights
    if candidate_flights is None and truth_flights is None:
        return candidate_rows, aligned_truth, ("sequence_id",)
    if candidate_flights is not None and truth_flights is not None:
        return candidate_rows, aligned_truth, ("sequence_id", "flight_id")

    if candidate_flights is not None:
        mapping = _single_flight_by_sequence(
            candidate_rows,
            candidate_flights,
            role="candidate",
        )
        aligned_truth["flight_id"] = aligned_truth["sequence_id"].astype(str).map(mapping)
    else:
        assert truth_flights is not None
        mapping = _single_flight_by_sequence(
            aligned_truth,
            truth_flights,
            role="truth",
        )
        candidate_rows["flight_id"] = candidate_rows["sequence_id"].astype(str).map(mapping)
    return candidate_rows, aligned_truth, ("sequence_id", "flight_id")


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


def _truth_by_scope(
    truth_rows: pd.DataFrame,
    scope_fields: tuple[str, ...],
) -> dict[tuple[str, ...], pd.DataFrame]:
    """Index normalized truth trajectories by physical scope."""

    if scope_fields == ("sequence_id",):
        return {
            (str(sequence_id),): group.sort_values("time_s", kind="mergesort").reset_index(
                drop=True
            )
            for sequence_id, group in truth_rows.groupby("sequence_id", sort=True)
        }
    return {
        tuple(str(value) for value in scope): group.sort_values(
            "time_s",
            kind="mergesort",
        ).reset_index(drop=True)
        for scope, group in truth_rows.groupby(list(scope_fields), sort=True)
    }


def build_candidate_oracle_attribution_tables(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    top_k_values: Sequence[int] = _IMPL._DEFAULT_TOP_K,
    score_column: str = "candidate_reservoir_score",
    fallback_score_column: str = "ranker_score",
    max_truth_time_delta_s: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return oracle-attribution tables with validated physical-flight scoping."""

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

    rows, truth_rows, scope_fields = _align_physical_scope(rows, truth_rows)
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
    truth_by_scope = _truth_by_scope(truth_rows, scope_fields)

    frame_records: list[dict[str, Any]] = []
    grouping_columns = [*scope_fields, "time_s"]
    for group_key, group in rows.groupby(grouping_columns, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        scope_key = tuple(str(value) for value in key_values[:-1])
        time_s = float(key_values[-1])
        scoped_truth = truth_by_scope.get(scope_key)
        if scoped_truth is None or scoped_truth.empty:
            continue
        truth_t = scoped_truth["time_s"].to_numpy(float)
        nearest_idx = int(np.argmin(np.abs(truth_t - time_s)))
        truth_dt = time_s - float(truth_t[nearest_idx])
        if abs(truth_dt) > max_delta:
            continue
        truth_xyz = scoped_truth.iloc[nearest_idx][["x_m", "y_m", "z_m"]].to_numpy(float)
        ranked = _deterministic_candidate_ranking(group)
        candidate_xyz = ranked[["x_m", "y_m", "z_m"]].to_numpy(float)
        distances = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
        best_pos = int(np.argmin(distances))
        best_row = ranked.iloc[best_pos]
        record: dict[str, Any] = {
            "sequence_id": scope_key[0],
            "time_s": time_s,
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
        if len(scope_key) > 1:
            record["flight_id"] = scope_key[1]
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
