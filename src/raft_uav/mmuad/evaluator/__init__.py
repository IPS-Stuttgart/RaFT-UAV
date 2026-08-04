"""Package wrapper that tightens public Track 5 result validation.

The legacy evaluator implementation lives in the sibling ``evaluator.py`` file.
This wrapper preserves public imports while overriding official-result
classification validation, normalizing local result sequence identifiers,
rejecting complex local trajectory values, retaining the final row for duplicate
truth timestamps, and using globally consistent one-to-one timestamp matching.
Official truth-file loading remains permissive so
existing local truth archives stay readable.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad import _submission_impl
from raft_uav.mmuad.submission import (
    OFFICIAL_TRACK5_CLASS_IDS,
    _validated_timestamp_tolerance,
)
from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment

_IMPL_PATH = Path(__file__).resolve().parent.parent / "evaluator.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._evaluator_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy MMUAD evaluator from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_VALIDATE_MMAUD_RESULTS_FRAME = _IMPL.validate_mmaud_results_frame
_ORIGINAL_EVALUATE_NEAREST_TIME_RESULTS = _IMPL._evaluate_nearest_time_results
_MISSING_SEQUENCE_ID_STRINGS = {"", "nan", "none", "<na>", "nat"}
_LOCAL_NUMERIC_RESULT_COLUMNS = (
    "timestamp",
    "time_s",
    "t",
    "x",
    "x_m",
    "y",
    "y_m",
    "z",
    "z_m",
    "score",
    "confidence",
)


def _parse_official_result_classification_cell(value: Any) -> int:
    class_id = _IMPL.parse_official_classification_cell(value)
    if class_id not in OFFICIAL_TRACK5_CLASS_IDS:
        allowed = ", ".join(str(item) for item in sorted(OFFICIAL_TRACK5_CLASS_IDS))
        raise ValueError(
            "official MMUAD Classification values must be one of "
            f"{{{allowed}}}; got {class_id!r}"
        )
    return class_id


def _parse_official_truth_classification_cell(value: Any) -> int:
    parser = getattr(
        _submission_impl,
        "_raft_uav_permissive_parse_official_classification_cell",
        _submission_impl.parse_official_classification_cell,
    )
    return parser(value)


def _normalized_official_track5_header(column: Any) -> str:
    return str(column).strip().lower()


def _validate_unique_official_track5_headers(columns: Any) -> None:
    """Reject ambiguous official fields before order-dependent lookup."""

    required = {
        _normalized_official_track5_header(column)
        for column in _submission_impl.OFFICIAL_UG2_RESULT_COLUMNS
    }
    physical_by_key: dict[str, list[str]] = {}
    for column in columns:
        key = _normalized_official_track5_header(column)
        if key in required:
            physical_by_key.setdefault(key, []).append(str(column))
    collisions = {
        key: physical
        for key, physical in physical_by_key.items()
        if len(physical) > 1
    }
    if collisions:
        details = ", ".join(
            f"{key!r}: {physical!r}"
            for key, physical in sorted(collisions.items())
        )
        raise ValueError(
            "official MMUAD results contain ambiguous columns after trimming "
            f"whitespace and ignoring case: {details}"
        )


def _official_track5_column_map(frame: pd.DataFrame) -> dict[str, Any]:
    """Map official Track 5 columns after rejecting normalized collisions."""

    _validate_unique_official_track5_headers(frame.columns)
    return {
        _normalized_official_track5_header(column): column
        for column in frame.columns
    }


def _has_official_track5_columns(frame: pd.DataFrame) -> bool:
    lower = set(_official_track5_column_map(frame))
    return {column.lower() for column in _submission_impl.OFFICIAL_UG2_RESULT_COLUMNS}.issubset(
        lower
    )


def _official_track5_results_to_local_frame(
    frame: pd.DataFrame,
    *,
    enforce_class_domain: bool = True,
) -> pd.DataFrame:
    lower_to_original = _official_track5_column_map(frame)
    sequence_col = lower_to_original["sequence"]
    timestamp_col = lower_to_original["timestamp"]
    position_col = lower_to_original["position"]
    classification_col = lower_to_original["classification"]
    sequences = [_IMPL.parse_official_sequence_cell(value) for value in frame[sequence_col]]
    timestamps = [_IMPL.parse_official_timestamp_cell(value) for value in frame[timestamp_col]]
    positions = [_IMPL.parse_official_position_cell(value) for value in frame[position_col]]
    class_parser = (
        _parse_official_result_classification_cell
        if enforce_class_domain
        else _parse_official_truth_classification_cell
    )
    classifications = [class_parser(value) for value in frame[classification_col]]
    xyz = pd.DataFrame(positions, columns=["x", "y", "z"], index=frame.index)
    return pd.DataFrame(
        {
            "sequence_id": sequences,
            "timestamp": timestamps,
            "x": xyz["x"],
            "y": xyz["y"],
            "z": xyz["z"],
            "uav_type": [str(value) for value in classifications],
            "score": 1.0,
        }
    )


def _official_track5_truth_to_rows(frame: pd.DataFrame) -> pd.DataFrame:
    local = _official_track5_results_to_local_frame(frame, enforce_class_domain=False)
    rows = local.rename(
        columns={
            "timestamp": "time_s",
            "x": "x_m",
            "y": "y_m",
            "z": "z_m",
            "uav_type": "class_name",
        }
    )
    return _IMPL.normalize_truth_columns(rows)


def _normalize_local_result_sequence_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize local result ids consistently with candidate and truth tables."""

    if "sequence_id" not in frame.columns:
        return frame
    rows = frame.copy()
    text = rows["sequence_id"].where(rows["sequence_id"].notna(), "default")
    text = text.astype(str).str.strip()
    missing = text.str.lower().isin(_MISSING_SEQUENCE_ID_STRINGS)
    rows["sequence_id"] = text.where(~missing, "default")
    return rows


def _is_complex_result_value(value: Any) -> bool:
    """Return whether one result cell contains a complex numeric payload."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    try:
        array = np.asanyarray(value)
    except (TypeError, ValueError):
        return False
    return bool(np.iscomplexobj(array))


def _reject_complex_local_result_values(frame: pd.DataFrame) -> None:
    """Reject complex values before float conversion can discard imaginary parts."""

    complex_rows = np.zeros(len(frame), dtype=bool)
    for column in _LOCAL_NUMERIC_RESULT_COLUMNS:
        if column not in frame.columns:
            continue
        complex_rows |= frame[column].map(_is_complex_result_value).to_numpy(dtype=bool)
    count = int(complex_rows.sum())
    if count:
        raise ValueError(f"mmaud_results contains {count} complex trajectory row(s)")


def validate_mmaud_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate results after normalizing local sequence identifiers."""

    normalized = _normalize_local_result_sequence_ids(frame)
    _reject_complex_local_result_values(normalized)
    validated = _ORIGINAL_VALIDATE_MMAUD_RESULTS_FRAME(normalized)
    dropped_count = len(normalized) - len(validated)
    if dropped_count:
        raise ValueError(
            "mmaud_results contains "
            f"{dropped_count} non-finite or non-numeric trajectory row(s)"
        )
    return validated


def _final_truth_rows_per_timestamp(truth_rows: pd.DataFrame) -> pd.DataFrame:
    """Retain the final truth row for each normalized sequence timestamp."""

    if truth_rows.empty:
        return truth_rows.copy()
    rows = truth_rows.copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["time_s"] = pd.to_numeric(rows["time_s"], errors="raise").astype(float)
    rows["_truth_row_order"] = np.arange(len(rows), dtype=np.int64)
    rows = rows.drop_duplicates(
        subset=["sequence_id", "time_s"],
        keep="last",
    )
    return (
        rows.sort_values(
            ["sequence_id", "time_s", "_truth_row_order"],
            kind="mergesort",
        )
        .drop(columns="_truth_row_order")
        .reset_index(drop=True)
    )


def _validated_max_time_delta_s(value: Any) -> float:
    """Return a finite, nonnegative, non-Boolean nearest-time gate."""

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
        max_delta_s = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(max_delta_s) or max_delta_s < 0.0:
        raise ValueError(error)
    return max_delta_s


def _evaluate_nearest_time_results(
    result_rows: pd.DataFrame,
    truth_rows: pd.DataFrame,
    *,
    class_map: dict[str, str],
    max_time_delta_s: Any,
) -> dict[str, Any]:
    """Evaluate nearest-time rows after validating the matching gate."""

    return _ORIGINAL_EVALUATE_NEAREST_TIME_RESULTS(
        result_rows,
        _final_truth_rows_per_timestamp(truth_rows),
        class_map=class_map,
        max_time_delta_s=_validated_max_time_delta_s(max_time_delta_s),
    )


def _read_physical_results_headers(source: Any) -> list[str]:
    """Read the unmangled first CSV row and restore reusable streams."""

    position: int | None = None
    if hasattr(source, "tell") and hasattr(source, "seek"):
        try:
            position = int(source.tell())
        except (OSError, TypeError, ValueError):
            position = None
    try:
        header = pd.read_csv(
            source,
            dtype=str,
            keep_default_na=False,
            header=None,
            nrows=1,
        )
    finally:
        if position is not None:
            source.seek(position)
    if header.empty:
        return []
    return [str(value) for value in header.iloc[0].tolist()]


def _read_results_csv_preserving_text(source: Any) -> pd.DataFrame:
    """Read evaluator CSV inputs without coercing ids or keeping padded headers."""

    _validate_unique_official_track5_headers(_read_physical_results_headers(source))
    try:
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    except TypeError:
        frame = pd.read_csv(source)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_mmaud_results_csv(path: Path) -> Any:
    frame = _read_results_csv_preserving_text(path)
    return _IMPL.ResultsFrame(_IMPL.validate_mmaud_results_frame(frame))


def _read_results_zip_csv(path: Path, *, member_name: str) -> pd.DataFrame:
    with _IMPL.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        selected = _IMPL._select_results_zip_member(infos, member_name=member_name)
        with archive.open(selected) as handle:
            return _read_results_csv_preserving_text(_IMPL.BytesIO(handle.read()))


def _evaluate_public_track5_timestamp_aligned(
    result_rows: pd.DataFrame,
    truth_rows: pd.DataFrame,
    *,
    class_map: dict[str, str],
    timestamp_tolerance_s: Any,
) -> dict[str, Any]:
    """Evaluate Track 5 rows with a globally optimal timestamp assignment."""

    tolerance = _validated_timestamp_tolerance(timestamp_tolerance_s)
    truth_rows = _final_truth_rows_per_timestamp(truth_rows)
    if truth_rows.empty:
        return _IMPL._empty_truth_evaluation(
            result_rows,
            metric_protocol="public_track5_timestamp_aligned",
            timestamp_tolerance_s=tolerance,
        )

    result_rows = result_rows.copy()
    result_rows["sequence_id"] = result_rows["sequence_id"].astype(str)
    used_result_indices: set[int] = set()
    error_records: list[dict[str, Any]] = []

    for sequence_id, seq_truth in truth_rows.groupby("sequence_id", sort=True):
        seq_truth = seq_truth.sort_values("time_s")
        seq_results = result_rows.loc[
            result_rows["sequence_id"] == str(sequence_id)
        ].sort_values("timestamp")
        assignment = optimal_timestamp_assignment(
            seq_truth["time_s"].to_numpy(float),
            seq_results["timestamp"].to_numpy(float),
            tolerance_s=tolerance,
        )
        for truth_position, (_, truth_row) in enumerate(seq_truth.iterrows()):
            prediction_position = assignment.get(truth_position)
            if prediction_position is None:
                error_records.append(_IMPL._missing_track5_prediction_row(truth_row))
                continue
            pred_index = int(seq_results.index[prediction_position])
            used_result_indices.add(pred_index)
            pred_row = seq_results.loc[pred_index]
            error_records.append(
                _IMPL._matched_track5_row(
                    pred_row,
                    truth_row,
                    class_map=class_map,
                )
            )

    error_records.extend(
        _IMPL._unused_track5_prediction_rows(
            result_rows,
            truth_rows,
            used_result_indices=used_result_indices,
            timestamp_tolerance_s=tolerance,
        )
    )
    errors = pd.DataFrame.from_records(error_records)
    matched = errors.loc[errors["matched"]].copy() if not errors.empty else pd.DataFrame()
    truth_count = int(len(truth_rows))
    prediction_count = int(len(result_rows))
    missing_count = _IMPL._reason_count(errors, "missing_prediction")
    extra_count = _IMPL._reason_count(errors, "extra_prediction")
    duplicate_count = _IMPL._reason_count(errors, "duplicate_prediction")
    blocking_reasons = _IMPL._track5_leaderboard_blocking_reasons(
        truth_count=truth_count,
        matched_count=int(len(matched)),
        missing_count=missing_count,
        extra_count=extra_count,
        duplicate_count=duplicate_count,
    )
    summary = {
        "metric_protocol": "public_track5_timestamp_aligned",
        "public_track5_metric": True,
        "closed_codabench_evaluator": False,
        "timestamp_tolerance_s": tolerance,
        "count": int(len(errors)),
        "truth_count": truth_count,
        "prediction_count": prediction_count,
        "matched_count": int(len(matched)),
        "missing_prediction_count": missing_count,
        "extra_prediction_count": extra_count,
        "duplicate_prediction_count": duplicate_count,
        "unmatched_count": int(len(errors) - len(matched)),
        "truth_coverage_fraction": float(len(matched) / truth_count) if truth_count else 0.0,
        "all_truth_timestamps_matched": int(len(matched)) == truth_count,
        "leaderboard_ready": not blocking_reasons,
        "score_valid_for_leaderboard": not blocking_reasons,
        "leaderboard_blocking_reasons": blocking_reasons,
        "pooled": _IMPL._error_summary(matched),
        "sequences": {},
    }
    summary["sequences"] = _IMPL._public_track5_sequence_summaries(
        errors,
        result_rows,
        truth_rows,
    )
    return {"summary": summary, "rows": errors}


_IMPL._has_official_track5_columns = _has_official_track5_columns
_IMPL._official_track5_results_to_local_frame = _official_track5_results_to_local_frame
_IMPL._official_track5_truth_to_rows = _official_track5_truth_to_rows
_IMPL.validate_mmaud_results_frame = validate_mmaud_results_frame
_IMPL._final_truth_rows_per_timestamp = _final_truth_rows_per_timestamp
_IMPL._validated_max_time_delta_s = _validated_max_time_delta_s
_IMPL._evaluate_nearest_time_results = _evaluate_nearest_time_results
_IMPL.load_mmaud_results_csv = load_mmaud_results_csv
_IMPL._read_results_zip_csv = _read_results_zip_csv
_IMPL._evaluate_public_track5_timestamp_aligned = _evaluate_public_track5_timestamp_aligned

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)

globals()["_parse_official_result_classification_cell"] = _parse_official_result_classification_cell
globals()["_parse_official_truth_classification_cell"] = _parse_official_truth_classification_cell
globals()["_normalized_official_track5_header"] = _normalized_official_track5_header
globals()["_validate_unique_official_track5_headers"] = _validate_unique_official_track5_headers
globals()["_official_track5_column_map"] = _official_track5_column_map
globals()["_normalize_local_result_sequence_ids"] = _normalize_local_result_sequence_ids
globals()["_is_complex_result_value"] = _is_complex_result_value
globals()["_reject_complex_local_result_values"] = _reject_complex_local_result_values
globals()["_final_truth_rows_per_timestamp"] = _final_truth_rows_per_timestamp
globals()["_validated_max_time_delta_s"] = _validated_max_time_delta_s
globals()["_evaluate_nearest_time_results"] = _evaluate_nearest_time_results
globals()["_read_physical_results_headers"] = _read_physical_results_headers
globals()["_read_results_csv_preserving_text"] = _read_results_csv_preserving_text
globals()["_evaluate_public_track5_timestamp_aligned"] = (
    _evaluate_public_track5_timestamp_aligned
)
__doc__ = _IMPL.__doc__
__all__ = [_name for _name in dir(_IMPL) if not _name.startswith("__")]
