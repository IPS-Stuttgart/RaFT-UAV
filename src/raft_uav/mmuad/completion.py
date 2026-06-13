"""Trajectory completion helpers for UG2+/MMUAD result tables.

The official challenge evaluates a position estimate at required timestamps.
This module provides a transparent local completion utility that resamples a
``mmaud_results.csv`` style trajectory to the timestamps present in a normalized
truth or template table. It is intended for local sanity checks and packaging;
it is not a claim of official Codabench evaluator equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import ResultsFrame, validate_mmaud_results_frame
from raft_uav.mmuad.schema import TruthFrame, normalize_truth_columns
from raft_uav.mmuad.submission import UG2_RESULT_COLUMNS


@dataclass(frozen=True)
class CompletionResult:
    """Completed result rows plus row-level diagnostics."""

    rows: pd.DataFrame
    diagnostics: pd.DataFrame


def complete_results_to_truth_timestamps(
    results: ResultsFrame | pd.DataFrame,
    truth_or_template: TruthFrame | pd.DataFrame,
    *,
    max_interpolation_gap_s: float = 1.0,
    extrapolation: str = "hold",
    default_score: float = 1.0,
) -> CompletionResult:
    """Resample results to the timestamps in a truth/template table."""

    if extrapolation not in {"hold", "nan"}:
        raise ValueError("extrapolation must be 'hold' or 'nan'")
    result_rows = _completion_result_rows(results)
    if isinstance(truth_or_template, TruthFrame):
        template = truth_or_template.rows.copy()
    else:
        template = normalize_truth_columns(truth_or_template)
    if template.empty:
        return CompletionResult(
            rows=pd.DataFrame(columns=UG2_RESULT_COLUMNS),
            diagnostics=pd.DataFrame(),
        )

    out_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for sequence_id, template_group in template.groupby("sequence_id", sort=True):
        seq_results = result_rows.loc[
            result_rows["sequence_id"] == str(sequence_id)
        ].sort_values("timestamp")
        if seq_results.empty:
            for _, template_row in template_group.sort_values("time_s").iterrows():
                diag_rows.append(
                    _diagnostic_row(
                        sequence_id=str(sequence_id),
                        timestamp=float(template_row["time_s"]),
                        method="missing_sequence_prediction",
                        source_left_time_s=np.nan,
                        source_right_time_s=np.nan,
                    )
                )
            continue

        times = seq_results["timestamp"].to_numpy(float)
        xyz = seq_results[["x", "y", "z"]].to_numpy(float)
        scores = seq_results["score"].to_numpy(float)
        uav_type = _mode_string(seq_results["uav_type"].astype(str))
        for _, template_row in template_group.sort_values("time_s").iterrows():
            timestamp = float(template_row["time_s"])
            completed = _complete_one(
                timestamp,
                times,
                xyz,
                scores,
                max_interpolation_gap_s=max_interpolation_gap_s,
                extrapolation=extrapolation,
            )
            if completed is None:
                diag_rows.append(
                    _diagnostic_row(
                        sequence_id=str(sequence_id),
                        timestamp=timestamp,
                        method="dropped_unfillable",
                        source_left_time_s=np.nan,
                        source_right_time_s=np.nan,
                    )
                )
                continue

            point, score, method, left_t, right_t = completed
            out_rows.append(
                {
                    "sequence_id": str(sequence_id),
                    "timestamp": timestamp,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "uav_type": uav_type,
                    "score": (
                        float(score) if np.isfinite(score) else float(default_score)
                    ),
                }
            )
            diag_rows.append(
                _diagnostic_row(
                    sequence_id=str(sequence_id),
                    timestamp=timestamp,
                    method=method,
                    source_left_time_s=left_t,
                    source_right_time_s=right_t,
                )
            )

    completed_rows = pd.DataFrame.from_records(out_rows, columns=UG2_RESULT_COLUMNS)
    diagnostics = pd.DataFrame.from_records(diag_rows)
    if not completed_rows.empty:
        completed_rows = validate_mmaud_results_frame(completed_rows)
    return CompletionResult(rows=completed_rows, diagnostics=diagnostics)


def _completion_result_rows(results: ResultsFrame | pd.DataFrame) -> pd.DataFrame:
    """Return validated result rows while preserving the no-prediction case."""

    raw_rows = results.rows if isinstance(results, ResultsFrame) else results
    if raw_rows.empty:
        return pd.DataFrame(columns=UG2_RESULT_COLUMNS)
    try:
        return validate_mmaud_results_frame(raw_rows)
    except ValueError as exc:
        if "contains no finite trajectory rows" not in str(exc):
            raise
    return pd.DataFrame(columns=UG2_RESULT_COLUMNS)


def completion_summary(
    result: CompletionResult, *, requested_count: int | None = None
) -> dict[str, Any]:
    """Return a compact summary for a completion result."""

    diagnostics = result.diagnostics
    method_counts = {}
    if not diagnostics.empty and "completion_method" in diagnostics.columns:
        method_counts = diagnostics["completion_method"].value_counts().to_dict()
    return {
        "requested_count": int(
            requested_count if requested_count is not None else len(diagnostics)
        ),
        "completed_count": int(len(result.rows)),
        "dropped_count": int(max(0, len(diagnostics) - len(result.rows))),
        "completion_method_counts": {
            str(key): int(value) for key, value in method_counts.items()
        },
    }


def _complete_one(
    timestamp: float,
    times: np.ndarray,
    xyz: np.ndarray,
    scores: np.ndarray,
    *,
    max_interpolation_gap_s: float,
    extrapolation: str,
) -> tuple[np.ndarray, float, str, float, float] | None:
    if len(times) == 1:
        if extrapolation == "hold":
            return xyz[0], scores[0], "hold_single", float(times[0]), float(times[0])
        return None

    idx_right = int(np.searchsorted(times, timestamp, side="left"))
    if idx_right < len(times) and abs(float(times[idx_right] - timestamp)) < 1e-9:
        return (
            xyz[idx_right],
            scores[idx_right],
            "exact",
            float(times[idx_right]),
            float(times[idx_right]),
        )

    idx_left = idx_right - 1
    if 0 <= idx_left and idx_right < len(times):
        left_t = float(times[idx_left])
        right_t = float(times[idx_right])
        gap = right_t - left_t
        if 0.0 < gap <= float(max_interpolation_gap_s):
            alpha = (timestamp - left_t) / gap
            point = (1.0 - alpha) * xyz[idx_left] + alpha * xyz[idx_right]
            score = (1.0 - alpha) * scores[idx_left] + alpha * scores[idx_right]
            return point, float(score), "interpolated", left_t, right_t

    if extrapolation == "hold":
        nearest_idx = int(np.argmin(np.abs(times - timestamp)))
        method = "hold_before" if times[nearest_idx] <= timestamp else "hold_after"
        return (
            xyz[nearest_idx],
            scores[nearest_idx],
            method,
            float(times[nearest_idx]),
            float(times[nearest_idx]),
        )
    return None


def _mode_string(values: pd.Series) -> str:
    if values.empty:
        return "unknown"
    mode = values.mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(values.iloc[0])


def _diagnostic_row(
    *,
    sequence_id: str,
    timestamp: float,
    method: str,
    source_left_time_s: float,
    source_right_time_s: float,
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "timestamp": float(timestamp),
        "completion_method": method,
        "source_left_time_s": source_left_time_s,
        "source_right_time_s": source_right_time_s,
    }
