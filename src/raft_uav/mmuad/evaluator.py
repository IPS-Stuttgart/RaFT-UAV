"""Local evaluator helpers for UG2+/MMUAD-style trajectory exports.

This module intentionally implements transparent local metrics, not the closed
Codabench runtime.  It validates ``mmaud_results.csv``-style files and can
evaluate either a nearest-time development diagnostic or the public Track 5
timestamp-aligned MSE/classification quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import json
import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import (
    TruthFrame,
    load_jsonable,
    normalize_time_column_aliases,
    normalize_truth_columns,
)
from raft_uav.mmuad.submission import (
    OFFICIAL_UG2_RESULT_COLUMNS,
    UG2_RESULT_COLUMNS,
    load_sequence_class_map,
    parse_official_classification_cell,
    parse_official_position_cell,
    parse_official_sequence_cell,
    parse_official_timestamp_cell,
)

_TRUTH_TYPE_COLUMNS = (
    "uav_type",
    "class_name",
    "class",
    "label",
    "category",
    "classification",
    "class_id",
    "uav_type_id",
    "type_id",
)


@dataclass(frozen=True)
class ResultsFrame:
    """Validated UG2-style result rows."""

    rows: pd.DataFrame


def load_mmaud_results_csv(path: Path) -> ResultsFrame:
    """Load and validate a ``mmaud_results.csv``-style file."""

    frame = pd.read_csv(path)
    return ResultsFrame(validate_mmaud_results_frame(frame))


def load_mmaud_results_file(path: Path) -> ResultsFrame:
    """Load result rows from a CSV file or a Codabench-style ZIP archive."""

    path = Path(path)
    if path.suffix.lower() == ".zip":
        return load_mmaud_results_zip(path)
    return load_mmaud_results_csv(path)


def load_mmaud_results_zip(
    path: Path,
    *,
    member_name: str = "mmaud_results.csv",
) -> ResultsFrame:
    """Load and validate ``mmaud_results.csv`` from a ZIP archive."""

    path = Path(path)
    with ZipFile(path) as archive:
        names = archive.namelist()
        if member_name not in names:
            raise ValueError(f"{path} does not contain {member_name!r}; members={names}")
        with archive.open(member_name) as handle:
            frame = pd.read_csv(BytesIO(handle.read()))
    return ResultsFrame(validate_mmaud_results_frame(frame))


def validate_mmaud_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized result frame or raise with actionable errors."""

    if _has_official_track5_columns(frame):
        frame = _official_track5_results_to_local_frame(frame)

    rename = {
        "time_s": "timestamp",
        "t": "timestamp",
        "x_m": "x",
        "y_m": "y",
        "z_m": "z",
        "class_name": "uav_type",
        "label": "uav_type",
        "confidence": "score",
    }
    rows = normalize_time_column_aliases(frame, target="timestamp")
    rows = rows.rename(
        columns={
            key: value
            for key, value in rename.items()
            if key in rows.columns and value not in rows.columns
        }
    ).copy()
    missing = set(UG2_RESULT_COLUMNS).difference(rows.columns)
    if missing:
        raise ValueError(f"mmaud_results rows missing columns: {sorted(missing)}")
    rows = rows[list(UG2_RESULT_COLUMNS)].copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["uav_type"] = rows["uav_type"].astype(str)
    for col in ("timestamp", "x", "y", "z", "score"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    finite = np.isfinite(rows[["timestamp", "x", "y", "z", "score"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy()
    if rows.empty:
        raise ValueError("mmaud_results contains no finite trajectory rows")
    return rows.sort_values(["sequence_id", "timestamp"]).reset_index(drop=True)


def _has_official_track5_columns(frame: pd.DataFrame) -> bool:
    lower = {str(column).lower() for column in frame.columns}
    return {column.lower() for column in OFFICIAL_UG2_RESULT_COLUMNS}.issubset(lower)


def _official_track5_results_to_local_frame(frame: pd.DataFrame) -> pd.DataFrame:
    lower_to_original = {str(column).lower(): column for column in frame.columns}
    sequence_col = lower_to_original["sequence"]
    timestamp_col = lower_to_original["timestamp"]
    position_col = lower_to_original["position"]
    classification_col = lower_to_original["classification"]
    sequences = [parse_official_sequence_cell(value) for value in frame[sequence_col]]
    timestamps = [parse_official_timestamp_cell(value) for value in frame[timestamp_col]]
    positions = [parse_official_position_cell(value) for value in frame[position_col]]
    classifications = [
        parse_official_classification_cell(value)
        for value in frame[classification_col]
    ]
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


def evaluate_mmaud_results(
    results: ResultsFrame | pd.DataFrame,
    truth: TruthFrame | pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
    metric_protocol: str = "nearest-time",
    timestamp_tolerance_s: float = 1.0e-6,
    class_map_csv: Path | None = None,
    class_map_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate result rows against normalized truth.

    ``nearest-time`` is an ADE/FDE-style development diagnostic.  ``public-track5``
    aligns predictions to the truth/template timestamps required by the public
    UG2+ Track 5 submission instructions and reports the public MSE and
    classification-accuracy quantities.  Neither mode claims closed Codabench
    runtime equivalence.
    """

    result_rows = (
        results.rows if isinstance(results, ResultsFrame) else validate_mmaud_results_frame(results)
    )
    truth_rows = truth.rows if isinstance(truth, TruthFrame) else normalize_truth_columns(truth)
    class_map_file = class_map_path if class_map_path is not None else class_map_csv
    class_map = load_sequence_class_map(class_map_file) if class_map_file is not None else {}
    protocol = _normalize_metric_protocol(metric_protocol)
    if protocol == "public_track5_timestamp_aligned":
        return _evaluate_public_track5_timestamp_aligned(
            result_rows,
            truth_rows,
            class_map=class_map,
            timestamp_tolerance_s=timestamp_tolerance_s,
        )
    return _evaluate_nearest_time_results(
        result_rows,
        truth_rows,
        class_map=class_map,
        max_time_delta_s=max_time_delta_s,
    )


def _normalize_metric_protocol(value: str) -> str:
    text = str(value).strip().lower().replace("_", "-")
    if text in {"nearest-time", "nearest", "nearest-truth"}:
        return "nearest_truth_with_time_gate"
    if text in {
        "public-track5",
        "track5-public",
        "official-track5",
        "public-track5-timestamp-aligned",
    }:
        return "public_track5_timestamp_aligned"
    raise ValueError(
        "metric_protocol must be 'nearest-time' or 'public-track5'; "
        f"got {value!r}"
    )


def _evaluate_nearest_time_results(
    result_rows: pd.DataFrame,
    truth_rows: pd.DataFrame,
    *,
    class_map: dict[str, str],
    max_time_delta_s: float,
) -> dict[str, Any]:
    if truth_rows.empty:
        return _empty_truth_evaluation(
            result_rows,
            metric_protocol="nearest_truth_with_time_gate",
            max_time_delta_s=max_time_delta_s,
        )
    truth_rows = truth_rows.copy()
    truth_rows["sequence_id"] = truth_rows["sequence_id"].astype(str)
    error_records: list[dict[str, Any]] = []
    for sequence_id, group in result_rows.groupby("sequence_id", sort=True):
        seq_truth = truth_rows.loc[truth_rows["sequence_id"] == sequence_id].sort_values("time_s")
        if seq_truth.empty:
            for _, row in group.iterrows():
                error_records.append(_unmatched_result_row(row, reason="missing_sequence_truth"))
            continue
        truth_t = seq_truth["time_s"].to_numpy(float)
        truth_xyz = seq_truth[["x_m", "y_m", "z_m"]].to_numpy(float)
        for _, row in group.iterrows():
            idx = int(np.argmin(np.abs(truth_t - float(row["timestamp"]))))
            dt = float(row["timestamp"] - truth_t[idx])
            if abs(dt) > float(max_time_delta_s):
                error_records.append(_unmatched_result_row(row, reason="time_delta_exceeded", dt_s=dt))
                continue
            pred = row[["x", "y", "z"]].to_numpy(float)
            err = pred - truth_xyz[idx]
            predicted_type = _canonical_type_label(row.get("uav_type", "")) or ""
            truth_type = _truth_type_for_row(seq_truth.iloc[idx], sequence_id, class_map)
            type_correct = (predicted_type == truth_type) if truth_type is not None else None
            error_records.append(
                {
                    "sequence_id": sequence_id,
                    "timestamp": float(row["timestamp"]),
                    "matched": True,
                    "time_delta_s": dt,
                    "error_2d_m": float(np.linalg.norm(err[:2])),
                    "error_3d_m": float(np.linalg.norm(err)),
                    "squared_error_3d_m2": float(np.dot(err, err)),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "truth_x_m": float(truth_xyz[idx, 0]),
                    "truth_y_m": float(truth_xyz[idx, 1]),
                    "truth_z_m": float(truth_xyz[idx, 2]),
                    "predicted_uav_type": predicted_type,
                    "truth_uav_type": truth_type,
                    "uav_type_correct": type_correct,
                }
            )
    errors = pd.DataFrame.from_records(error_records)
    matched = errors.loc[errors["matched"]].copy() if not errors.empty else pd.DataFrame()
    summary = {
        "metric_protocol": "nearest_truth_with_time_gate",
        "max_time_delta_s": float(max_time_delta_s),
        "count": int(len(errors)),
        "matched_count": int(len(matched)),
        "unmatched_count": int(len(errors) - len(matched)),
        "pooled": _error_summary(matched),
        "sequences": {},
    }
    if not matched.empty:
        for sequence_id, group in matched.groupby("sequence_id", sort=True):
            summary["sequences"][str(sequence_id)] = _error_summary(group)
    return {"summary": summary, "rows": errors}


def _evaluate_public_track5_timestamp_aligned(
    result_rows: pd.DataFrame,
    truth_rows: pd.DataFrame,
    *,
    class_map: dict[str, str],
    timestamp_tolerance_s: float,
) -> dict[str, Any]:
    if timestamp_tolerance_s < 0.0:
        raise ValueError("timestamp_tolerance_s must be non-negative")
    if truth_rows.empty:
        return _empty_truth_evaluation(
            result_rows,
            metric_protocol="public_track5_timestamp_aligned",
            timestamp_tolerance_s=timestamp_tolerance_s,
        )

    result_rows = result_rows.copy()
    truth_rows = truth_rows.copy()
    result_rows["sequence_id"] = result_rows["sequence_id"].astype(str)
    truth_rows["sequence_id"] = truth_rows["sequence_id"].astype(str)
    used_result_indices: set[int] = set()
    error_records: list[dict[str, Any]] = []

    for sequence_id, seq_truth in truth_rows.groupby("sequence_id", sort=True):
        seq_truth = seq_truth.sort_values("time_s")
        seq_results = result_rows.loc[
            result_rows["sequence_id"] == str(sequence_id)
        ].sort_values("timestamp")
        result_times = seq_results["timestamp"].to_numpy(float)
        for _, truth_row in seq_truth.iterrows():
            truth_time = float(truth_row["time_s"])
            if seq_results.empty:
                error_records.append(_missing_track5_prediction_row(truth_row))
                continue
            candidates = np.flatnonzero(np.abs(result_times - truth_time) <= timestamp_tolerance_s)
            if len(candidates) == 0:
                error_records.append(_missing_track5_prediction_row(truth_row))
                continue
            nearest_pos = int(candidates[np.argmin(np.abs(result_times[candidates] - truth_time))])
            pred_index = int(seq_results.index[nearest_pos])
            used_result_indices.add(pred_index)
            pred_row = seq_results.loc[pred_index]
            error_records.append(
                _matched_track5_row(
                    pred_row,
                    truth_row,
                    class_map=class_map,
                )
            )

    error_records.extend(
        _unused_track5_prediction_rows(
            result_rows,
            truth_rows,
            used_result_indices=used_result_indices,
            timestamp_tolerance_s=timestamp_tolerance_s,
        )
    )
    errors = pd.DataFrame.from_records(error_records)
    matched = errors.loc[errors["matched"]].copy() if not errors.empty else pd.DataFrame()
    truth_count = int(len(truth_rows))
    prediction_count = int(len(result_rows))
    missing_count = _reason_count(errors, "missing_prediction")
    extra_count = _reason_count(errors, "extra_prediction")
    duplicate_count = _reason_count(errors, "duplicate_prediction")
    blocking_reasons = _track5_leaderboard_blocking_reasons(
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
        "timestamp_tolerance_s": float(timestamp_tolerance_s),
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
        "pooled": _error_summary(matched),
        "sequences": {},
    }
    for sequence_id, seq_truth in truth_rows.groupby("sequence_id", sort=True):
        group = errors.loc[errors["sequence_id"].astype(str) == str(sequence_id)]
        seq_matched = group.loc[group["matched"]].copy() if not group.empty else pd.DataFrame()
        seq_truth_count = int(len(seq_truth))
        seq_prediction_count = int(
            (result_rows["sequence_id"].astype(str) == str(sequence_id)).sum()
        )
        seq_summary = _error_summary(seq_matched)
        seq_missing_count = _reason_count(group, "missing_prediction")
        seq_extra_count = _reason_count(group, "extra_prediction")
        seq_duplicate_count = _reason_count(group, "duplicate_prediction")
        seq_blocking_reasons = _track5_leaderboard_blocking_reasons(
            truth_count=seq_truth_count,
            matched_count=int(len(seq_matched)),
            missing_count=seq_missing_count,
            extra_count=seq_extra_count,
            duplicate_count=seq_duplicate_count,
        )
        seq_summary.update(
            {
                "truth_count": seq_truth_count,
                "prediction_count": seq_prediction_count,
                "matched_count": int(len(seq_matched)),
                "missing_prediction_count": seq_missing_count,
                "extra_prediction_count": seq_extra_count,
                "duplicate_prediction_count": seq_duplicate_count,
                "truth_coverage_fraction": (
                    float(len(seq_matched) / seq_truth_count) if seq_truth_count else 0.0
                ),
                "all_truth_timestamps_matched": int(len(seq_matched)) == seq_truth_count,
                "leaderboard_ready": not seq_blocking_reasons,
                "score_valid_for_leaderboard": not seq_blocking_reasons,
                "leaderboard_blocking_reasons": seq_blocking_reasons,
            }
        )
        summary["sequences"][str(sequence_id)] = seq_summary
    return {"summary": summary, "rows": errors}


def _track5_leaderboard_blocking_reasons(
    *,
    truth_count: int,
    matched_count: int,
    missing_count: int,
    extra_count: int,
    duplicate_count: int,
) -> list[str]:
    reasons: list[str] = []
    if truth_count == 0:
        reasons.append("no_truth_timestamps")
    if matched_count != truth_count:
        reasons.append("not_all_truth_timestamps_matched")
    if missing_count:
        reasons.append("missing_predictions")
    if extra_count:
        reasons.append("extra_predictions")
    if duplicate_count:
        reasons.append("duplicate_predictions")
    return reasons


def write_evaluation_artifacts(
    result: dict[str, Any],
    *,
    summary_json: Path,
    rows_csv: Path | None = None,
) -> dict[str, str]:
    """Write evaluator summary/rows and return created paths."""

    summary_json = Path(summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(load_jsonable(result["summary"]), indent=2),
        encoding="utf-8",
    )
    paths = {"evaluation_json": str(summary_json)}
    if rows_csv is not None:
        rows_csv = Path(rows_csv)
        rows_csv.parent.mkdir(parents=True, exist_ok=True)
        result["rows"].to_csv(rows_csv, index=False)
        paths["evaluation_rows_csv"] = str(rows_csv)
    return paths


def _matched_track5_row(
    pred_row: pd.Series,
    truth_row: pd.Series,
    *,
    class_map: dict[str, str],
) -> dict[str, Any]:
    pred = pred_row[["x", "y", "z"]].to_numpy(float)
    truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
    err = pred - truth_xyz
    sequence_id = str(truth_row["sequence_id"])
    predicted_type = _canonical_type_label(pred_row.get("uav_type", "")) or ""
    truth_type = _truth_type_for_row(truth_row, sequence_id, class_map)
    type_correct = (predicted_type == truth_type) if truth_type is not None else None
    return {
        "sequence_id": sequence_id,
        "timestamp": float(pred_row["timestamp"]),
        "truth_timestamp": float(truth_row["time_s"]),
        "matched": True,
        "unmatched_reason": "",
        "time_delta_s": float(pred_row["timestamp"] - truth_row["time_s"]),
        "error_2d_m": float(np.linalg.norm(err[:2])),
        "error_3d_m": float(np.linalg.norm(err)),
        "squared_error_3d_m2": float(np.dot(err, err)),
        "x": float(pred_row["x"]),
        "y": float(pred_row["y"]),
        "z": float(pred_row["z"]),
        "truth_x_m": float(truth_xyz[0]),
        "truth_y_m": float(truth_xyz[1]),
        "truth_z_m": float(truth_xyz[2]),
        "predicted_uav_type": predicted_type,
        "truth_uav_type": truth_type,
        "uav_type_correct": type_correct,
    }


def _missing_track5_prediction_row(truth_row: pd.Series) -> dict[str, Any]:
    return {
        "sequence_id": str(truth_row.get("sequence_id", "")),
        "timestamp": np.nan,
        "truth_timestamp": float(truth_row.get("time_s", np.nan)),
        "matched": False,
        "unmatched_reason": "missing_prediction",
        "time_delta_s": np.nan,
        "error_2d_m": np.nan,
        "error_3d_m": np.nan,
        "squared_error_3d_m2": np.nan,
        "x": np.nan,
        "y": np.nan,
        "z": np.nan,
        "truth_x_m": float(truth_row.get("x_m", np.nan)),
        "truth_y_m": float(truth_row.get("y_m", np.nan)),
        "truth_z_m": float(truth_row.get("z_m", np.nan)),
        "predicted_uav_type": "",
        "truth_uav_type": None,
        "uav_type_correct": None,
    }


def _unused_track5_prediction_rows(
    result_rows: pd.DataFrame,
    truth_rows: pd.DataFrame,
    *,
    used_result_indices: set[int],
    timestamp_tolerance_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    truth_by_sequence = {
        str(sequence_id): group["time_s"].to_numpy(float)
        for sequence_id, group in truth_rows.groupby("sequence_id", sort=True)
    }
    for index, pred_row in result_rows.iterrows():
        if int(index) in used_result_indices:
            continue
        sequence_id = str(pred_row.get("sequence_id", ""))
        predicted_type = _canonical_type_label(pred_row.get("uav_type", "")) or ""
        truth_times = truth_by_sequence.get(sequence_id, np.asarray([], dtype=float))
        if truth_times.size:
            nearest_delta = float(
                np.min(np.abs(truth_times - float(pred_row["timestamp"])))
            )
            reason = (
                "duplicate_prediction"
                if nearest_delta <= float(timestamp_tolerance_s)
                else "extra_prediction"
            )
            truth_timestamp = float(
                truth_times[int(np.argmin(np.abs(truth_times - float(pred_row["timestamp"]))))]
            )
        else:
            nearest_delta = np.nan
            reason = "extra_prediction"
            truth_timestamp = np.nan
        rows.append(
            {
                "sequence_id": sequence_id,
                "timestamp": float(pred_row.get("timestamp", np.nan)),
                "truth_timestamp": truth_timestamp,
                "matched": False,
                "unmatched_reason": reason,
                "time_delta_s": (
                    float(pred_row["timestamp"] - truth_timestamp)
                    if np.isfinite(truth_timestamp)
                    else nearest_delta
                ),
                "error_2d_m": np.nan,
                "error_3d_m": np.nan,
                "squared_error_3d_m2": np.nan,
                "x": float(pred_row.get("x", np.nan)),
                "y": float(pred_row.get("y", np.nan)),
                "z": float(pred_row.get("z", np.nan)),
                "truth_x_m": np.nan,
                "truth_y_m": np.nan,
                "truth_z_m": np.nan,
                "predicted_uav_type": predicted_type,
                "truth_uav_type": None,
                "uav_type_correct": None,
            }
        )
    return rows


def _reason_count(frame: pd.DataFrame, reason: str) -> int:
    if frame.empty or "unmatched_reason" not in frame.columns:
        return 0
    return int((frame["unmatched_reason"].astype(str) == reason).sum())


def _empty_truth_evaluation(
    result_rows: pd.DataFrame,
    *,
    metric_protocol: str,
    max_time_delta_s: float | None = None,
    timestamp_tolerance_s: float | None = None,
) -> dict[str, Any]:
    """Return the standard evaluator payload when no truth rows are available."""

    errors = pd.DataFrame.from_records(
        [
            _unmatched_result_row(row, reason="empty_truth")
            for _, row in result_rows.iterrows()
        ]
    )
    summary = {
        "metric_protocol": metric_protocol,
        "count": int(len(errors)),
        "truth_count": 0,
        "prediction_count": int(len(result_rows)),
        "matched_count": 0,
        "unmatched_count": int(len(errors)),
        "pooled": _error_summary(pd.DataFrame()),
        "sequences": {},
    }
    if max_time_delta_s is not None:
        summary["max_time_delta_s"] = float(max_time_delta_s)
    if timestamp_tolerance_s is not None:
        summary["timestamp_tolerance_s"] = float(timestamp_tolerance_s)
    return {"summary": summary, "rows": errors}


def _unmatched_result_row(row: pd.Series, *, reason: str, dt_s: float | None = None) -> dict[str, Any]:
    predicted_type = _canonical_type_label(row.get("uav_type", "")) or ""
    return {
        "sequence_id": str(row.get("sequence_id", "")),
        "timestamp": float(row.get("timestamp", np.nan)),
        "matched": False,
        "unmatched_reason": reason,
        "time_delta_s": dt_s,
        "error_2d_m": np.nan,
        "error_3d_m": np.nan,
        "squared_error_3d_m2": np.nan,
        "x": float(row.get("x", np.nan)),
        "y": float(row.get("y", np.nan)),
        "z": float(row.get("z", np.nan)),
        "predicted_uav_type": predicted_type,
        "truth_uav_type": None,
        "uav_type_correct": None,
    }


def _error_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"count": 0}
    err3 = frame["error_3d_m"].to_numpy(float)
    err2 = frame["error_2d_m"].to_numpy(float)
    out = {
        "count": int(len(frame)),
        "mean_3d_m": float(np.nanmean(err3)),
        "rmse_3d_m": float(np.sqrt(np.nanmean(err3**2))),
        "pose_mse_loss_m2": float(np.nanmean(err3**2)),
        "mean_square_loss_m2": float(np.nanmean(err3**2)),
        "p50_3d_m": float(np.nanpercentile(err3, 50.0)),
        "p95_3d_m": float(np.nanpercentile(err3, 95.0)),
        "max_3d_m": float(np.nanmax(err3)),
        "mean_2d_m": float(np.nanmean(err2)),
        "p95_2d_m": float(np.nanpercentile(err2, 95.0)),
        "max_2d_m": float(np.nanmax(err2)),
        "mean_abs_time_delta_s": float(np.nanmean(np.abs(frame["time_delta_s"].to_numpy(float)))),
    }
    if "uav_type_correct" in frame.columns:
        typed = frame.loc[frame["uav_type_correct"].notna()].copy()
        if not typed.empty:
            correct = typed["uav_type_correct"].astype(bool).to_numpy()
            out["uav_type_count"] = int(len(correct))
            out["uav_type_accuracy"] = float(np.mean(correct))
            out["classification_accuracy"] = float(np.mean(correct))
    return out


def _truth_type_for_row(
    truth_row: pd.Series, sequence_id: str, class_map: dict[str, str]
) -> str | None:
    for column in _TRUTH_TYPE_COLUMNS:
        if column in truth_row.index and pd.notna(truth_row[column]):
            return _canonical_type_label(truth_row[column])
    return _canonical_type_label(class_map.get(str(sequence_id)))


def _canonical_type_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except TypeError:
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text
