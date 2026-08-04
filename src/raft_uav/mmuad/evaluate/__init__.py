"""Maintained MMUAD submission evaluation overrides.

The legacy implementation remains in the sibling ``evaluate.py`` module. This
package preserves the public import path while rejecting ambiguous
case/whitespace-equivalent submission headers, rejecting malformed required
numeric rows before the legacy loader can silently drop them, replacing
nearest-time matching with a cardinality-first one-to-one assignment that is
independent of CSV row order, retaining authoritative final same-time truth
snapshots, and normalizing serialized match flags before metrics are summarized.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_IMPL_PATH = Path(__file__).resolve().parent.parent / "evaluate.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._evaluate_impl",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise ImportError(f"cannot load MMUAD evaluation implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_normalize_submission_sequence_ids = _IMPL._normalize_submission_sequence_ids
_valid_track_id_text = _IMPL._valid_track_id_text
normalize_truth_columns = _IMPL.normalize_truth_columns
_unmatched_prediction_row = _IMPL._unmatched_prediction_row
_track_ids = _IMPL._track_ids
_should_restrict_to_track_id = _IMPL._should_restrict_to_track_id
_truth_track_id = _IMPL._truth_track_id
_ORIGINAL_LOAD_SUBMISSION_CSV = _IMPL.load_submission_csv
_ORIGINAL_METRICS_FROM_MATCHES = _IMPL.metrics_from_matches
_REQUIRED_SUBMISSION_NUMERIC_COLUMNS = ("time_s", "x_m", "y_m", "z_m")
_TRUE_MATCH_TEXT = frozenset({"true", "t", "yes", "y", "1", "1.0"})
_FALSE_MATCH_TEXT = frozenset(
    {
        "false",
        "f",
        "no",
        "n",
        "0",
        "0.0",
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }
)


def _normalized_submission_header(value: Any) -> str:
    """Return a case- and whitespace-insensitive submission header key."""

    return str(value).strip().casefold()


def _validate_unique_submission_headers(columns: Any) -> None:
    """Reject physical CSV headers that become indistinguishable."""

    column_list = list(columns)
    normalized = [_normalized_submission_header(column) for column in column_list]
    duplicate_mask = pd.Index(normalized).duplicated(keep=False)
    if not bool(duplicate_mask.any()):
        return
    ambiguous = sorted(
        {
            str(column)
            for column, duplicated in zip(column_list, duplicate_mask)
            if duplicated
        },
        key=lambda column: (_normalized_submission_header(column), column),
    )
    rendered = ", ".join(repr(column) for column in ambiguous)
    raise ValueError(
        "submission has ambiguous columns after trimming whitespace "
        f"and ignoring case: {rendered}"
    )


def _read_physical_submission_headers(path: Path) -> list[str]:
    """Read the unmangled CSV header before pandas deduplicates names."""

    header = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        header=None,
        nrows=1,
    )
    if header.empty:
        return []
    return [str(value) for value in header.iloc[0].tolist()]


def _submission_rows_before_numeric_coercion(path: Path) -> pd.DataFrame:
    """Return alias-normalized CSV rows before numeric coercion or filtering."""

    frame = _IMPL.normalize_time_column_aliases(
        pd.read_csv(path, dtype=str, keep_default_na=False),
        target="time_s",
    )
    return _IMPL._rename_submission_aliases(frame)


def _invalid_submission_row_summary(index: pd.Index, invalid: np.ndarray) -> str:
    """Return a compact list of invalid original row labels."""

    positions = np.flatnonzero(np.asarray(invalid, dtype=bool))
    labels = [repr(index[int(position)]) for position in positions[:5]]
    suffix = ", ..." if len(positions) > 5 else ""
    return ", ".join(labels) + suffix


def _validate_submission_numeric_rows(path: Path) -> None:
    """Reject malformed required numeric rows before the legacy row filter runs."""

    frame = _submission_rows_before_numeric_coercion(path)
    missing = set(_REQUIRED_SUBMISSION_NUMERIC_COLUMNS).difference(frame.columns)
    if missing:
        return
    for column in _REQUIRED_SUBMISSION_NUMERIC_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        invalid = ~np.isfinite(numeric)
        if not bool(invalid.any()):
            continue
        count = int(np.count_nonzero(invalid))
        sample = _invalid_submission_row_summary(frame.index, invalid)
        raise ValueError(
            f"{path} contains {count} invalid submission {column} row(s) "
            f"at index/indices {sample}"
        )


def load_submission_csv(path: Path) -> pd.DataFrame:
    """Load a stable submission after validating headers and required rows."""

    _validate_unique_submission_headers(_read_physical_submission_headers(path))
    _validate_submission_numeric_rows(path)
    return _ORIGINAL_LOAD_SUBMISSION_CSV(path)


def _validated_max_time_delta_s(value: Any) -> float:
    """Return a finite, nonnegative, non-Boolean scalar time gate."""

    error = "max_time_delta_s must be a finite nonnegative real scalar"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        time_delta_s = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(time_delta_s) or time_delta_s < 0.0:
        raise ValueError(error)
    return time_delta_s


def _authoritative_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    """Retain the final normalized same-time truth row per identifiable target."""

    raw = pd.DataFrame(truth).copy()
    order_column = "_submission_eval_truth_input_order"
    while order_column in raw.columns:
        order_column = f"_{order_column}"
    raw[order_column] = np.arange(len(raw), dtype=np.int64)
    rows = normalize_truth_columns(raw)
    if rows.empty:
        return rows.drop(columns=[order_column], errors="ignore")

    time_key = ["sequence_id", "time_s"]
    if "track_id" not in rows.columns:
        return (
            rows.sort_values([*time_key, order_column], kind="mergesort")
            .drop_duplicates(time_key, keep="last")
            .drop(columns=[order_column])
            .reset_index(drop=True)
        )

    track_key_column = "_submission_eval_truth_track_key"
    while track_key_column in rows.columns:
        track_key_column = f"_{track_key_column}"
    rows[track_key_column] = rows["track_id"].map(_valid_track_id_text)
    identified = rows.loc[rows[track_key_column].notna()]
    anonymous = rows.loc[rows[track_key_column].isna()]
    identified = (
        identified.sort_values(
            [*time_key, track_key_column, order_column],
            kind="mergesort",
        )
        .drop_duplicates([*time_key, track_key_column], keep="last")
    )
    return (
        pd.concat([identified, anonymous], ignore_index=True)
        .sort_values([*time_key, order_column], kind="mergesort")
        .drop(columns=[order_column, track_key_column])
        .reset_index(drop=True)
    )


def match_submission_to_truth(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
) -> pd.DataFrame:
    """Match predictions to truth with cardinality-first optimal assignment.

    Existing sequence and track-ID gating semantics are retained. Within each
    sequence, the assignment first maximizes the number of truth rows matched
    inside the time gate and then minimizes total absolute timestamp error.
    """

    max_time_delta_s = _validated_max_time_delta_s(max_time_delta_s)
    if submission.empty:
        return pd.DataFrame()
    submission = submission.copy()
    if "sequence_id" not in submission.columns:
        submission["sequence_id"] = "default"
    else:
        submission["sequence_id"] = _normalize_submission_sequence_ids(
            submission["sequence_id"]
        )
    if "track_id" not in submission.columns:
        submission["track_id"] = ""
    else:
        submission["track_id"] = submission["track_id"].map(
            lambda value: _valid_track_id_text(value) or ""
        )
    truth = _authoritative_truth_rows(truth)
    if "track_id" in truth.columns:
        truth["track_id"] = truth["track_id"].map(
            lambda value: _valid_track_id_text(value) or ""
        )

    rows: list[dict[str, Any]] = []
    for sequence_id, pred_seq in submission.groupby("sequence_id", sort=True):
        truth_seq = truth.loc[truth["sequence_id"] == sequence_id].copy()
        if truth_seq.empty:
            rows.extend(
                _unmatched_prediction_row(pred, reason="missing_sequence_truth")
                for _, pred in pred_seq.iterrows()
            )
            continue

        pred_seq = pred_seq.reset_index(drop=True)
        truth_seq = truth_seq.reset_index(drop=True)
        truth_track_ids = (
            _track_ids(truth_seq) if "track_id" in truth_seq.columns else set()
        )
        submitted_track_ids = (
            _track_ids(pred_seq) if "track_id" in pred_seq.columns else set()
        )
        restrict_to_track_id = _should_restrict_to_track_id(
            truth_track_ids,
            submitted_track_ids,
        ) or (len(truth_track_ids) > 1 and not submitted_track_ids)
        assignments, eligible = _optimal_time_assignment(
            pred_seq,
            truth_seq,
            restrict_to_track_id=restrict_to_track_id,
            max_time_delta_s=max_time_delta_s,
        )
        truth_track_values = (
            truth_seq["track_id"].map(_valid_track_id_text).to_numpy(dtype=object)
            if "track_id" in truth_seq.columns
            else np.full(len(truth_seq), None, dtype=object)
        )

        for pred_position, pred in pred_seq.iterrows():
            pred_track_id = _valid_track_id_text(pred.get("track_id", ""))
            if restrict_to_track_id and (
                pred_track_id is None or pred_track_id not in truth_track_ids
            ):
                rows.append(_unmatched_prediction_row(pred, reason="track_id_mismatch"))
                continue

            truth_position = assignments.get(int(pred_position))
            if truth_position is not None:
                rows.append(
                    _matched_prediction_row(
                        sequence_id=sequence_id,
                        prediction=pred,
                        truth=truth_seq.iloc[truth_position],
                    )
                )
                continue

            candidate_mask = np.ones(len(truth_seq), dtype=bool)
            if restrict_to_track_id:
                candidate_mask = truth_track_values == pred_track_id
            if not bool(candidate_mask.any()):
                reason = "missing_track_truth"
            elif bool(eligible[int(pred_position)].any()):
                reason = "duplicate_truth_match"
            else:
                reason = "time_gate"
            rows.append(_unmatched_prediction_row(pred, reason=reason))
    return pd.DataFrame.from_records(rows)


def _optimal_time_assignment(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    restrict_to_track_id: bool,
    max_time_delta_s: float,
) -> tuple[dict[int, int], np.ndarray]:
    """Return a cardinality-first one-to-one assignment and eligibility mask."""

    max_time_delta_s = _validated_max_time_delta_s(max_time_delta_s)
    pred_times = pd.to_numeric(predictions["time_s"], errors="coerce").to_numpy(float)
    truth_times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(float)
    time_delta = np.abs(pred_times[:, np.newaxis] - truth_times[np.newaxis, :])
    eligible = np.isfinite(time_delta) & (time_delta <= max_time_delta_s)

    if restrict_to_track_id:
        pred_track_ids = (
            predictions["track_id"].map(_valid_track_id_text).to_numpy(dtype=object)
        )
        truth_track_ids = (
            truth["track_id"].map(_valid_track_id_text).to_numpy(dtype=object)
        )
        eligible &= pred_track_ids[:, np.newaxis] == truth_track_ids[np.newaxis, :]

    if not bool(eligible.any()):
        return {}, eligible

    max_matches = min(len(predictions), len(truth))
    distance_weight = 0.5 / float(max_matches + 1)
    distance_scale = max(float(np.max(time_delta[eligible])), 1.0)
    costs = np.full(
        (len(predictions), len(truth) + len(predictions)),
        1.0,
        dtype=float,
    )
    costs[:, : len(truth)] = np.where(
        eligible,
        distance_weight * time_delta / distance_scale,
        2.0,
    )
    pred_positions, assignment_columns = linear_sum_assignment(costs)
    assignments = {
        int(pred_position): int(truth_position)
        for pred_position, truth_position in zip(pred_positions, assignment_columns)
        if truth_position < len(truth) and eligible[pred_position, truth_position]
    }
    return assignments, eligible


def _matched_prediction_row(
    *,
    sequence_id: str,
    prediction: pd.Series,
    truth: pd.Series,
) -> dict[str, Any]:
    error = np.array(
        [
            float(prediction["x_m"]) - float(truth["x_m"]),
            float(prediction["y_m"]) - float(truth["y_m"]),
            float(prediction["z_m"]) - float(truth["z_m"]),
        ],
        dtype=float,
    )
    return {
        "sequence_id": sequence_id,
        "time_s": float(prediction["time_s"]),
        "track_id": _valid_track_id_text(prediction.get("track_id", "uav0"))
        or "uav0",
        "truth_time_s": float(truth["time_s"]),
        "truth_track_id": _truth_track_id(truth),
        "time_delta_s": abs(float(truth["time_s"]) - float(prediction["time_s"])),
        "matched": True,
        "unmatched_reason": "",
        "error_2d_m": float(np.linalg.norm(error[:2])),
        "error_3d_m": float(np.linalg.norm(error)),
        "vertical_error_m": float(error[2]),
    }


def _normalized_match_flags(values: Any) -> pd.Series:
    """Parse native and serialized Boolean match diagnostics explicitly."""

    series = pd.Series(values, copy=False)
    if series.empty:
        return pd.Series(index=series.index, dtype=bool)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean").fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    truthy = (text.isin(_TRUE_MATCH_TEXT) | numeric.eq(1.0)).fillna(False)
    falsy = (
        series.isna() | text.isin(_FALSE_MATCH_TEXT) | numeric.eq(0.0)
    ).fillna(False)
    invalid = ~(truthy | falsy)
    if bool(invalid.any()):
        invalid_indices = invalid[invalid].index.tolist()
        invalid_values = series.loc[invalid_indices].tolist()
        raise ValueError(
            "matched contains invalid Boolean values at rows "
            f"{invalid_indices}: {invalid_values}"
        )
    return truthy.astype(bool)


def metrics_from_matches(
    matches: pd.DataFrame,
    *,
    submission: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[str, Any]:
    """Compute metrics against the same authoritative truth used for matching."""

    normalized = matches.copy()
    if "matched" in normalized.columns:
        normalized["matched"] = _normalized_match_flags(normalized["matched"])
    return _ORIGINAL_METRICS_FROM_MATCHES(
        normalized,
        submission=submission,
        truth=_authoritative_truth_rows(truth),
    )


_IMPL.load_submission_csv = load_submission_csv
_IMPL._authoritative_truth_rows = _authoritative_truth_rows
_IMPL.match_submission_to_truth = match_submission_to_truth
_IMPL.metrics_from_matches = metrics_from_matches

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalized_submission_header"] = _normalized_submission_header
globals()["_validate_unique_submission_headers"] = _validate_unique_submission_headers
globals()["_read_physical_submission_headers"] = _read_physical_submission_headers
globals()["_submission_rows_before_numeric_coercion"] = (
    _submission_rows_before_numeric_coercion
)
globals()["_invalid_submission_row_summary"] = _invalid_submission_row_summary
globals()["_validate_submission_numeric_rows"] = _validate_submission_numeric_rows
globals()["load_submission_csv"] = load_submission_csv
globals()["_validated_max_time_delta_s"] = _validated_max_time_delta_s
globals()["_authoritative_truth_rows"] = _authoritative_truth_rows
globals()["match_submission_to_truth"] = match_submission_to_truth
globals()["_optimal_time_assignment"] = _optimal_time_assignment
globals()["_matched_prediction_row"] = _matched_prediction_row
globals()["_normalized_match_flags"] = _normalized_match_flags
globals()["metrics_from_matches"] = metrics_from_matches

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
