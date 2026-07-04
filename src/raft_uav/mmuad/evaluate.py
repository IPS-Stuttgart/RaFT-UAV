"""Submission evaluation helpers for MMUAD-style trajectory exports.

This module computes repository-level trajectory diagnostics for the stable
RaFT-UAV MMUAD interchange format.  It is not an official UG2+ evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import evaluate_mmaud_results, load_evaluation_truth_file
from raft_uav.mmuad.io import load_truth_file
from raft_uav.mmuad.schema import normalize_time_column_aliases, normalize_truth_columns

_SUBMISSION_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sequence_id": ("sequence", "seq", "scene", "scene_id", "clip", "clip_id"),
    "track_id": ("track", "id", "object_id", "cluster_id", "instance_id"),
    "x_m": ("x", "east_m", "pos_x", "center_x", "cx"),
    "y_m": ("y", "north_m", "pos_y", "center_y", "cy"),
    "z_m": ("z", "up_m", "pos_z", "center_z", "cz"),
    "uav_type": (
        "classification",
        "class_id",
        "class_name",
        "class",
        "label",
        "category",
        "type",
        "uav_class",
    ),
    "score": ("confidence", "probability"),
}
_MISSING_TRACK_ID_TOKENS = {"", "nan", "none", "<na>", "nat"}


def load_submission_csv(path: Path) -> pd.DataFrame:
    """Load the stable RaFT-UAV MMUAD trajectory CSV export."""

    frame = normalize_time_column_aliases(pd.read_csv(path), target="time_s")
    frame = _rename_submission_aliases(frame)
    missing = {"sequence_id", "time_s", "x_m", "y_m", "z_m"}.difference(frame.columns)
    if missing:
        raise ValueError(f"submission missing required columns: {sorted(missing)}")
    if "track_id" not in frame.columns:
        frame["track_id"] = "uav0"
    if "score" not in frame.columns:
        frame["score"] = 1.0
    for col in ("time_s", "x_m", "y_m", "z_m", "score"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["sequence_id"] = frame["sequence_id"].astype(str)
    frame["track_id"] = frame["track_id"].astype(str)
    if "uav_type" in frame.columns:
        frame["uav_type"] = frame["uav_type"].astype(str)
    return frame.loc[np.isfinite(frame[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()


def _rename_submission_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize submission CSV columns using case-insensitive canonical names and aliases."""

    lower_to_original = {str(col).lower(): col for col in frame.columns}
    rename: dict[Any, str] = {}
    for canonical, aliases in _SUBMISSION_COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue
        original = lower_to_original.get(canonical.lower())
        if original is not None:
            rename[original] = canonical
            continue
        for alias in aliases:
            original = lower_to_original.get(alias.lower())
            if original is not None:
                rename[original] = canonical
                break
    return frame.rename(columns=rename)


def evaluate_submission_csv(
    submission_csv: Path,
    truth_file: Path,
    *,
    max_time_delta_s: float = 0.5,
    metric_protocol: str = "nearest-time",
    timestamp_tolerance_s: float = 1.0e-6,
    class_map_path: Path | None = None,
    class_map_csv: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a stable trajectory CSV against normalized truth rows."""

    submission = load_submission_csv(submission_csv)
    if _normalize_submission_metric_protocol(metric_protocol) == "public_track5":
        class_map_file = class_map_path if class_map_path is not None else class_map_csv
        result = evaluate_mmaud_results(
            _submission_to_results_frame(submission),
            load_evaluation_truth_file(truth_file),
            metric_protocol="public-track5",
            timestamp_tolerance_s=timestamp_tolerance_s,
            class_map_path=class_map_file,
        )
        return _stable_submission_public_track5_summary(result["summary"])

    truth = load_truth_file(truth_file).rows
    matched = match_submission_to_truth(
        submission,
        truth,
        max_time_delta_s=max_time_delta_s,
    )
    return metrics_from_matches(matched, submission=submission, truth=truth)


def _normalize_submission_metric_protocol(value: str) -> str:
    text = str(value).strip().lower().replace("_", "-")
    if text in {"nearest-time", "nearest", "nearest-truth"}:
        return "nearest_time"
    if text in {
        "public-track5",
        "track5-public",
        "official-track5",
        "public-track5-timestamp-aligned",
    }:
        return "public_track5"
    raise ValueError(
        "metric_protocol must be 'nearest-time' or 'public-track5'; "
        f"got {value!r}"
    )


def _submission_to_results_frame(submission: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "sequence_id": submission["sequence_id"].astype(str),
            "timestamp": pd.to_numeric(submission["time_s"], errors="coerce"),
            "x": pd.to_numeric(submission["x_m"], errors="coerce"),
            "y": pd.to_numeric(submission["y_m"], errors="coerce"),
            "z": pd.to_numeric(submission["z_m"], errors="coerce"),
            "uav_type": (
                submission["uav_type"].astype(str)
                if "uav_type" in submission.columns
                else "unknown"
            ),
            "score": pd.to_numeric(submission["score"], errors="coerce"),
        }
    )
    return rows


def _stable_submission_public_track5_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(summary)
    out["schema"] = "raft-uav-mmuad-submission-eval-v1"
    out["stable_submission_csv"] = True
    out["official_ug2_metric"] = False
    out["closed_codabench_evaluator"] = False
    out["codabench_upload_ready"] = False
    out["leaderboard_ready"] = False
    reasons = [
        str(reason)
        for reason in out.get("leaderboard_blocking_reasons", [])
        if str(reason)
    ]
    reason = "stable_submission_csv_not_official_track5_package"
    if reason not in reasons:
        reasons.append(reason)
    out["leaderboard_blocking_reasons"] = reasons
    return out


def match_submission_to_truth(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
) -> pd.DataFrame:
    """Nearest-time match submission rows to truth rows within each sequence.

    When truth has a ``track_id`` column and the submitted track IDs overlap, the
    match is restricted to the same track ID.  Otherwise a single-UAV style
    sequence-level nearest-time match is used.  Each truth row can be consumed at
    most once so duplicate predictions are counted as unmatched predictions
    instead of inflating the matched trajectory count.
    """

    if truth.empty or submission.empty:
        return pd.DataFrame()
    submission = submission.copy()
    if "sequence_id" not in submission.columns:
        submission["sequence_id"] = "default"
    else:
        submission["sequence_id"] = submission["sequence_id"].astype(str)
    if "track_id" in submission.columns:
        submission["track_id"] = submission["track_id"].astype(str)
    truth = normalize_truth_columns(truth)
    if "track_id" in truth.columns:
        truth["track_id"] = truth["track_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for sequence_id, pred_seq in submission.groupby("sequence_id", sort=True):
        truth_seq = truth.loc[truth["sequence_id"] == sequence_id].copy()
        if truth_seq.empty:
            for _, pred in pred_seq.iterrows():
                rows.append(_unmatched_prediction_row(pred, reason="missing_sequence_truth"))
            continue
        truth_track_ids = _track_ids(truth_seq) if "track_id" in truth_seq.columns else set()
        submitted_track_ids = _track_ids(pred_seq) if "track_id" in pred_seq.columns else set()
        restrict_to_track_id = bool(
            truth_track_ids and truth_track_ids.intersection(submitted_track_ids)
        )
        used_truth_indices: set[Any] = set()
        for _, pred in pred_seq.iterrows():
            candidate_truth = truth_seq
            if restrict_to_track_id:
                pred_track_id = _valid_track_id_text(pred.get("track_id", ""))
                if pred_track_id is None or pred_track_id not in truth_track_ids:
                    rows.append(_unmatched_prediction_row(pred, reason="track_id_mismatch"))
                    continue
                candidate_truth = truth_seq.loc[
                    truth_seq["track_id"].map(_valid_track_id_text) == pred_track_id
                ]
            if candidate_truth.empty:
                rows.append(_unmatched_prediction_row(pred, reason="missing_track_truth"))
                continue

            time_delta_all = (
                candidate_truth["time_s"].astype(float) - float(pred["time_s"])
            ).abs()
            nearest_idx = time_delta_all.idxmin()
            nearest_dt = float(time_delta_all.loc[nearest_idx])

            available_truth = candidate_truth.loc[
                ~candidate_truth.index.isin(used_truth_indices)
            ]
            if available_truth.empty:
                reason = (
                    "duplicate_truth_match"
                    if nearest_dt <= float(max_time_delta_s)
                    else "time_gate"
                )
                rows.append(_unmatched_prediction_row(pred, reason=reason))
                continue

            time_delta_available = (
                available_truth["time_s"].astype(float) - float(pred["time_s"])
            ).abs()
            idx = time_delta_available.idxmin()
            gt = available_truth.loc[idx]
            dt = float(time_delta_available.loc[idx])
            if dt > float(max_time_delta_s):
                reason = (
                    "duplicate_truth_match"
                    if nearest_idx in used_truth_indices
                    and nearest_dt <= float(max_time_delta_s)
                    else "time_gate"
                )
                rows.append(_unmatched_prediction_row(pred, reason=reason))
                continue
            error = np.array(
                [
                    float(pred["x_m"]) - float(gt["x_m"]),
                    float(pred["y_m"]) - float(gt["y_m"]),
                    float(pred["z_m"]) - float(gt["z_m"]),
                ],
                dtype=float,
            )
            used_truth_indices.add(idx)
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": float(pred["time_s"]),
                    "track_id": str(pred.get("track_id", "uav0")),
                    "truth_time_s": float(gt["time_s"]),
                    "truth_track_id": _truth_track_id(gt),
                    "time_delta_s": dt,
                    "matched": True,
                    "unmatched_reason": "",
                    "error_2d_m": float(np.linalg.norm(error[:2])),
                    "error_3d_m": float(np.linalg.norm(error)),
                    "vertical_error_m": float(error[2]),
                }
            )
    return pd.DataFrame.from_records(rows)


def metrics_from_matches(
    matches: pd.DataFrame,
    *,
    submission: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[str, Any]:
    """Compute pooled/per-sequence submission diagnostics from match rows."""

    matched = matches.loc[matches.get("matched", False)].copy() if not matches.empty else matches
    pooled = _error_metrics(matched)
    truth_count = int(len(truth))
    prediction_count = int(len(submission))
    matched_count = int(len(matched))
    covered_truth_count = _covered_truth_count(matched, truth)
    pooled.update(
        {
            "truth_count": truth_count,
            "prediction_count": prediction_count,
            "matched_count": matched_count,
            "unmatched_prediction_count": int(prediction_count - matched_count),
            "covered_truth_count": covered_truth_count,
            "truth_coverage_fraction": (
                float(covered_truth_count / truth_count) if truth_count else 0.0
            ),
        }
    )
    by_sequence: dict[str, Any] = {}
    for sequence_id in _metric_sequence_ids(matches, submission=submission, truth=truth):
        seq_truth = _rows_for_sequence(truth, sequence_id)
        seq_pred = _rows_for_sequence(submission, sequence_id)
        group = _rows_for_sequence(matches, sequence_id)
        seq_matched = (
            group.loc[group["matched"].astype(bool)]
            if not group.empty and "matched" in group.columns
            else group.iloc[0:0].copy()
        )
        seq_covered_truth_count = _covered_truth_count(seq_matched, seq_truth)
        metrics = _error_metrics(seq_matched)
        metrics.update(
            {
                "truth_count": int(len(seq_truth)),
                "prediction_count": int(len(seq_pred)),
                "matched_count": int(len(seq_matched)),
                "covered_truth_count": seq_covered_truth_count,
                "truth_coverage_fraction": (
                    float(seq_covered_truth_count / len(seq_truth))
                    if len(seq_truth)
                    else 0.0
                ),
            }
        )
        by_sequence[str(sequence_id)] = metrics
    return {
        "schema": "raft-uav-mmuad-submission-eval-v1",
        "official_ug2_metric": False,
        "pooled": pooled,
        "sequences": by_sequence,
    }


def _track_ids(frame: pd.DataFrame) -> set[str]:
    track_ids: set[str] = set()
    for value in frame["track_id"]:
        track_id = _valid_track_id_text(value)
        if track_id is not None:
            track_ids.add(track_id)
    return track_ids


def _valid_track_id_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except TypeError:
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    text = str(value).strip()
    if text.lower() in _MISSING_TRACK_ID_TOKENS:
        return None
    return text


def _metric_sequence_ids(
    matches: pd.DataFrame,
    *,
    submission: pd.DataFrame,
    truth: pd.DataFrame,
) -> list[str]:
    """Return all sequence ids with truth, predictions, or match diagnostics."""

    sequence_ids: set[str] = set()
    for frame in (truth, submission, matches):
        if frame.empty or "sequence_id" not in frame.columns:
            continue
        sequence_ids.update(frame["sequence_id"].dropna().astype(str))
    return sorted(sequence_ids)


def _rows_for_sequence(frame: pd.DataFrame, sequence_id: str) -> pd.DataFrame:
    if frame.empty or "sequence_id" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["sequence_id"].astype(str) == str(sequence_id)].copy()


def _covered_truth_count(matched: pd.DataFrame, truth: pd.DataFrame) -> int:
    """Count unique truth samples covered by at least one matched prediction."""

    if matched.empty:
        return 0
    key_columns = ["sequence_id", "truth_time_s"]
    if "track_id" in truth.columns and "truth_track_id" in matched.columns:
        truth_track_ids = _track_ids(truth)
        matched_truth_track_ids = matched["truth_track_id"].map(_valid_track_id_text)
        if truth_track_ids and matched_truth_track_ids.isin(truth_track_ids).any():
            key_columns.append("truth_track_id")
    if any(column not in matched.columns for column in key_columns):
        return int(len(matched))
    covered = matched[key_columns].copy()
    truth_times = pd.to_numeric(covered["truth_time_s"], errors="coerce")
    covered = covered.loc[truth_times.notna()]
    return int(covered.drop_duplicates().shape[0])


def _error_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "error_3d_m" not in frame.columns:
        return {"count": 0}
    err3 = frame["error_3d_m"].to_numpy(float)
    err2 = frame["error_2d_m"].to_numpy(float)
    out = {
        "count": int(np.isfinite(err3).sum()),
        "mean_3d_m": _nanmean(err3),
        "rmse_3d_m": float(np.sqrt(np.nanmean(err3**2))) if np.isfinite(err3).any() else None,
        "p95_3d_m": _nanpercentile(err3, 95.0),
        "max_3d_m": _nanmax(err3),
        "ade_3d_m": _nanmean(err3),
        "fde_3d_m": _final_error(frame, "error_3d_m"),
        "mean_2d_m": _nanmean(err2),
        "p95_2d_m": _nanpercentile(err2, 95.0),
        "max_2d_m": _nanmax(err2),
        "fde_2d_m": _final_error(frame, "error_2d_m"),
    }
    return out


def _final_error(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    if "time_s" not in frame.columns:
        return float(values[np.flatnonzero(finite)[-1]])
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    timed = finite & np.isfinite(times)
    if not timed.any():
        return float(values[np.flatnonzero(finite)[-1]])
    timed_indices = np.flatnonzero(timed)
    latest_time = float(np.max(times[timed_indices]))
    latest_indices = timed_indices[times[timed_indices] == latest_time]
    return float(values[latest_indices[-1]])


def _truth_track_id(row: pd.Series) -> str:
    if "track_id" not in row.index:
        return ""
    return _valid_track_id_text(row["track_id"]) or ""


def _unmatched_prediction_row(pred: pd.Series, *, reason: str) -> dict[str, Any]:
    return {
        "sequence_id": str(pred.get("sequence_id", "default")),
        "time_s": float(pred.get("time_s", np.nan)),
        "track_id": str(pred.get("track_id", "uav0")),
        "truth_time_s": np.nan,
        "truth_track_id": "",
        "time_delta_s": np.nan,
        "matched": False,
        "unmatched_reason": reason,
        "error_2d_m": np.nan,
        "error_3d_m": np.nan,
        "vertical_error_m": np.nan,
    }


def _nanmean(values: np.ndarray) -> float | None:
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def _nanmax(values: np.ndarray) -> float | None:
    return float(np.nanmax(values)) if np.isfinite(values).any() else None


def _nanpercentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.nanpercentile(values, percentile)) if np.isfinite(values).any() else None
