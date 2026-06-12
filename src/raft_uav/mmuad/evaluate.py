"""Submission evaluation helpers for MMUAD-style trajectory exports.

This module computes repository-level trajectory diagnostics for the stable
RaFT-UAV MMUAD interchange format.  It is not an official UG2+ evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.io import load_truth_file
from raft_uav.mmuad.schema import normalize_truth_columns


def load_submission_csv(path: Path) -> pd.DataFrame:
    """Load the stable RaFT-UAV MMUAD trajectory CSV export."""

    frame = pd.read_csv(path)
    aliases = {
        "timestamp_s": "time_s",
        "track": "track_id",
        "id": "track_id",
        "x": "x_m",
        "y": "y_m",
        "z": "z_m",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
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
    return frame.loc[np.isfinite(frame[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()


def evaluate_submission_csv(
    submission_csv: Path,
    truth_file: Path,
    *,
    max_time_delta_s: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a stable trajectory CSV against normalized truth rows."""

    submission = load_submission_csv(submission_csv)
    truth = load_truth_file(truth_file).rows
    matched = match_submission_to_truth(
        submission,
        truth,
        max_time_delta_s=max_time_delta_s,
    )
    return metrics_from_matches(matched, submission=submission, truth=truth)


def match_submission_to_truth(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
) -> pd.DataFrame:
    """Nearest-time match submission rows to truth rows within each sequence.

    When truth has a ``track_id`` column and the submitted track IDs overlap, the
    match is restricted to the same track ID.  Otherwise a single-UAV style
    sequence-level nearest-time match is used.
    """

    if truth.empty or submission.empty:
        return pd.DataFrame()
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
        for _, pred in pred_seq.iterrows():
            candidate_truth = truth_seq
            if "track_id" in truth_seq.columns and str(pred.get("track_id", "")) in set(truth_seq["track_id"]):
                candidate_truth = truth_seq.loc[truth_seq["track_id"] == str(pred["track_id"])]
            idx = (candidate_truth["time_s"].astype(float) - float(pred["time_s"])).abs().idxmin()
            gt = candidate_truth.loc[idx]
            dt = abs(float(gt["time_s"]) - float(pred["time_s"]))
            if dt > float(max_time_delta_s):
                rows.append(_unmatched_prediction_row(pred, reason="time_gate"))
                continue
            error = np.array(
                [
                    float(pred["x_m"]) - float(gt["x_m"]),
                    float(pred["y_m"]) - float(gt["y_m"]),
                    float(pred["z_m"]) - float(gt["z_m"]),
                ],
                dtype=float,
            )
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": float(pred["time_s"]),
                    "track_id": str(pred.get("track_id", "uav0")),
                    "truth_time_s": float(gt["time_s"]),
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
    pooled.update(
        {
            "truth_count": truth_count,
            "prediction_count": prediction_count,
            "matched_count": matched_count,
            "unmatched_prediction_count": int(prediction_count - matched_count),
            "truth_coverage_fraction": float(matched_count / truth_count) if truth_count else 0.0,
        }
    )
    by_sequence: dict[str, Any] = {}
    if not matches.empty:
        for sequence_id, group in matches.groupby("sequence_id", sort=True):
            seq_truth = truth.loc[truth["sequence_id"].astype(str) == str(sequence_id)]
            seq_pred = submission.loc[submission["sequence_id"].astype(str) == str(sequence_id)]
            seq_matched = group.loc[group["matched"]]
            metrics = _error_metrics(seq_matched)
            metrics.update(
                {
                    "truth_count": int(len(seq_truth)),
                    "prediction_count": int(len(seq_pred)),
                    "matched_count": int(len(seq_matched)),
                }
            )
            by_sequence[str(sequence_id)] = metrics
    return {
        "schema": "raft-uav-mmuad-submission-eval-v1",
        "official_ug2_metric": False,
        "pooled": pooled,
        "sequences": by_sequence,
    }


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
        "fde_3d_m": float(err3[np.isfinite(err3)][-1]) if np.isfinite(err3).any() else None,
        "mean_2d_m": _nanmean(err2),
        "p95_2d_m": _nanpercentile(err2, 95.0),
        "max_2d_m": _nanmax(err2),
    }
    return out


def _unmatched_prediction_row(pred: pd.Series, *, reason: str) -> dict[str, Any]:
    return {
        "sequence_id": str(pred.get("sequence_id", "default")),
        "time_s": float(pred.get("time_s", np.nan)),
        "track_id": str(pred.get("track_id", "uav0")),
        "truth_time_s": np.nan,
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
