"""Local evaluator helpers for UG2+/MMUAD-style trajectory exports.

This module intentionally implements a transparent local metric, not the
closed Codabench evaluator.  It validates ``mmaud_results.csv``-style files and
compares them to normalized truth with nearest-time association so that tracker
outputs can be sanity-checked before official submission packaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import (
    TruthFrame,
    load_jsonable,
    normalize_time_column_aliases,
    normalize_truth_columns,
)
from raft_uav.mmuad.submission import UG2_RESULT_COLUMNS, load_sequence_class_map


@dataclass(frozen=True)
class ResultsFrame:
    """Validated UG2-style result rows."""

    rows: pd.DataFrame


def load_mmaud_results_csv(path: Path) -> ResultsFrame:
    """Load and validate a ``mmaud_results.csv``-style file."""

    frame = pd.read_csv(path)
    return ResultsFrame(validate_mmaud_results_frame(frame))


def validate_mmaud_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized result frame or raise with actionable errors."""

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


def evaluate_mmaud_results(
    results: ResultsFrame | pd.DataFrame,
    truth: TruthFrame | pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
    class_map_csv: Path | None = None,
) -> dict[str, Any]:
    """Evaluate result rows against normalized truth by nearest timestamp.

    The returned values are ADE/FDE-style diagnostics for local development.
    They are not a claim of official UG2+ evaluator equivalence.
    """

    result_rows = results.rows if isinstance(results, ResultsFrame) else validate_mmaud_results_frame(results)
    truth_rows = truth.rows if isinstance(truth, TruthFrame) else normalize_truth_columns(truth)
    class_map = load_sequence_class_map(class_map_csv) if class_map_csv is not None else {}
    if truth_rows.empty:
        return _empty_truth_evaluation(result_rows, max_time_delta_s=max_time_delta_s)
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
            predicted_type = str(row.get("uav_type", ""))
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


def _empty_truth_evaluation(
    result_rows: pd.DataFrame,
    *,
    max_time_delta_s: float,
) -> dict[str, Any]:
    """Return the standard evaluator payload when no truth rows are available."""

    errors = pd.DataFrame.from_records(
        [
            _unmatched_result_row(row, reason="empty_truth")
            for _, row in result_rows.iterrows()
        ]
    )
    summary = {
        "metric_protocol": "nearest_truth_with_time_gate",
        "max_time_delta_s": float(max_time_delta_s),
        "count": int(len(errors)),
        "matched_count": 0,
        "unmatched_count": int(len(errors)),
        "pooled": _error_summary(pd.DataFrame()),
        "sequences": {},
    }
    return {"summary": summary, "rows": errors}


def _unmatched_result_row(row: pd.Series, *, reason: str, dt_s: float | None = None) -> dict[str, Any]:
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
        "predicted_uav_type": str(row.get("uav_type", "")),
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
    return out


def _truth_type_for_row(
    truth_row: pd.Series, sequence_id: str, class_map: dict[str, str]
) -> str | None:
    for column in ("uav_type", "class_name", "class", "label", "category"):
        if column in truth_row.index and pd.notna(truth_row[column]):
            return str(truth_row[column])
    return class_map.get(str(sequence_id))
