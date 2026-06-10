"""Submission and metric-export helpers for MMUAD-style experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUBMISSION_COLUMNS = (
    "sequence_id",
    "time_s",
    "track_id",
    "x_m",
    "y_m",
    "z_m",
    "score",
)


def estimates_to_submission_frame(
    estimates: pd.DataFrame,
    *,
    track_id: str = "raft_uav_pp",
) -> pd.DataFrame:
    """Convert tracker estimates into a simple challenge-ready trajectory table."""

    if estimates.empty:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)
    rows = pd.DataFrame(
        {
            "sequence_id": estimates.get("sequence_id", "default"),
            "time_s": estimates["time_s"].astype(float),
            "track_id": track_id,
            "x_m": estimates["state_x_m"].astype(float),
            "y_m": estimates["state_y_m"].astype(float),
            "z_m": estimates["state_z_m"].astype(float),
            "score": 1.0,
        }
    )
    return (
        rows[list(SUBMISSION_COLUMNS)]
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


def write_submission_csv(
    estimates: pd.DataFrame,
    path: Path,
    *,
    track_id: str = "raft_uav_pp",
) -> Path:
    """Write a simple single-UAV trajectory submission CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    estimates_to_submission_frame(estimates, track_id=track_id).to_csv(path, index=False)
    return path


def write_submission_json(
    estimates: pd.DataFrame,
    path: Path,
    *,
    track_id: str = "raft_uav_pp",
) -> Path:
    """Write a simple JSON trajectory export.

    This is not the official UG2+ upload schema; it is a stable interchange file
    for downstream conversion once the official evaluator/submission format is
    available.
    """

    frame = estimates_to_submission_frame(estimates, track_id=track_id)
    payload: dict[str, Any] = {
        "schema": "raft-uav-mmuad-single-uav-trajectory-v1",
        "track_id": track_id,
        "sequences": {},
    }
    for sequence_id, group in frame.groupby("sequence_id", sort=True):
        payload["sequences"][str(sequence_id)] = group.drop(
            columns=["sequence_id"]
        ).to_dict(orient="records")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def compute_trajectory_metrics(estimates: pd.DataFrame) -> dict[str, Any]:
    """Compute extra trajectory metrics when truth-error columns are present."""

    if estimates.empty or "error_3d_m" not in estimates.columns:
        return {"count": int(len(estimates))}
    rows: dict[str, Any] = {"sequences": {}, "pooled": _metrics_for_frame(estimates)}
    if "sequence_id" in estimates.columns:
        for sequence_id, group in estimates.groupby("sequence_id", sort=True):
            rows["sequences"][str(sequence_id)] = _metrics_for_frame(group)
    return rows


def _metrics_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    err = frame["error_3d_m"].to_numpy(float)
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return {"count": 0}
    out = {
        "count": int(finite.size),
        "mean_3d_m": float(np.mean(finite)),
        "rmse_3d_m": float(np.sqrt(np.mean(finite**2))),
        "p95_3d_m": float(np.percentile(finite, 95.0)),
        "max_3d_m": float(np.max(finite)),
        "ade_3d_m": float(np.mean(finite)),
        "fde_3d_m": float(finite[-1]),
    }
    if "error_2d_m" in frame.columns:
        err2 = frame["error_2d_m"].to_numpy(float)
        finite2 = err2[np.isfinite(err2)]
        if finite2.size:
            out.update(
                {
                    "mean_2d_m": float(np.mean(finite2)),
                    "p95_2d_m": float(np.percentile(finite2, 95.0)),
                    "max_2d_m": float(np.max(finite2)),
                    "ade_2d_m": float(np.mean(finite2)),
                    "fde_2d_m": float(finite2[-1]),
                }
            )
    return out
