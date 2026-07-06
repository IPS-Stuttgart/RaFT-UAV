"""Trajectory completion helpers for UG2+/MMUAD result tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import ResultsFrame, validate_mmaud_results_frame
from raft_uav.mmuad.schema import TruthFrame, normalize_time_column_aliases, normalize_truth_columns
from raft_uav.mmuad.submission import UG2_RESULT_COLUMNS


_MISSING_TEXT_STRINGS = {"", "nan", "none", "<na>", "nat"}


@dataclass(frozen=True)
class CompletionResult:
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
    if extrapolation not in {"hold", "nan"}:
        raise ValueError("extrapolation must be 'hold' or 'nan'")
    max_gap_s = _normalize_max_interpolation_gap_s(max_interpolation_gap_s)
    result_rows = _completion_result_rows(results)
    template = _completion_template_rows(truth_or_template)
    if template.empty:
        return CompletionResult(pd.DataFrame(columns=UG2_RESULT_COLUMNS), pd.DataFrame())

    out_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for sequence_id, template_group in template.groupby("sequence_id", sort=True):
        seq = str(sequence_id)
        seq_results = result_rows.loc[result_rows["sequence_id"] == seq].sort_values("timestamp")
        if seq_results.empty:
            for _, row in template_group.sort_values("time_s").iterrows():
                diag_rows.append(
                    _diagnostic_row(
                        seq,
                        float(row["time_s"]),
                        "missing_sequence_prediction",
                        np.nan,
                        np.nan,
                    )
                )
            continue
        times = seq_results["timestamp"].to_numpy(float)
        xyz = seq_results[["x", "y", "z"]].to_numpy(float)
        scores = seq_results["score"].to_numpy(float)
        uav_type = _mode_string(seq_results["uav_type"])
        for _, row in template_group.sort_values("time_s").iterrows():
            timestamp = float(row["time_s"])
            completed = _complete_one(timestamp, times, xyz, scores, max_gap_s, extrapolation)
            if completed is None:
                diag_rows.append(_diagnostic_row(seq, timestamp, "dropped_unfillable", np.nan, np.nan))
                continue
            point, score, method, left_t, right_t = completed
            out_rows.append(
                {
                    "sequence_id": seq,
                    "timestamp": timestamp,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "uav_type": uav_type,
                    "score": float(score) if np.isfinite(score) else float(default_score),
                }
            )
            diag_rows.append(_diagnostic_row(seq, timestamp, method, left_t, right_t))
    rows = pd.DataFrame.from_records(out_rows, columns=UG2_RESULT_COLUMNS)
    if not rows.empty:
        rows = validate_mmaud_results_frame(rows)
    return CompletionResult(rows, pd.DataFrame.from_records(diag_rows))


def _completion_result_rows(results: ResultsFrame | pd.DataFrame) -> pd.DataFrame:
    raw = results.rows if isinstance(results, ResultsFrame) else results
    if raw.empty:
        return pd.DataFrame(columns=UG2_RESULT_COLUMNS)
    try:
        return validate_mmaud_results_frame(raw)
    except ValueError as exc:
        if "contains no finite trajectory rows" not in str(exc):
            raise
        return pd.DataFrame(columns=UG2_RESULT_COLUMNS)


def _completion_template_rows(truth_or_template: TruthFrame | pd.DataFrame) -> pd.DataFrame:
    if isinstance(truth_or_template, TruthFrame):
        template = truth_or_template.rows.copy()
    else:
        try:
            template = normalize_truth_columns(truth_or_template)
        except ValueError as exc:
            template = _timestamp_only_template_rows(truth_or_template, exc)
    if template.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s"])
    rows = template[["sequence_id", "time_s"]].copy()
    rows["sequence_id"] = rows["sequence_id"].fillna("default").astype(str).str.strip().replace("", "default")
    rows["time_s"] = pd.to_numeric(rows["time_s"], errors="coerce")
    return (
        rows.loc[np.isfinite(rows["time_s"].to_numpy(float))]
        .drop_duplicates()
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


def _timestamp_only_template_rows(truth_or_template: pd.DataFrame, cause: ValueError) -> pd.DataFrame:
    rows = normalize_time_column_aliases(pd.DataFrame(truth_or_template).copy(), target="time_s")
    lower = {str(column).lower(): column for column in rows.columns}
    rename: dict[Any, str] = {}
    for alias in ("sequence_id", "sequence", "seq", "scene", "scene_id", "id", "name"):
        if alias in lower:
            rename[lower[alias]] = "sequence_id"
            break
    if "timestamp" in lower and "time_s" not in rows.columns:
        rename[lower["timestamp"]] = "time_s"
    rows = rows.rename(columns=rename)
    missing = {"sequence_id", "time_s"}.difference(rows.columns)
    if missing:
        raise ValueError(f"completion template missing columns: {sorted(missing)}") from cause
    return rows[["sequence_id", "time_s"]].copy()


def completion_summary(result: CompletionResult, *, requested_count: int | None = None) -> dict[str, Any]:
    diagnostics = result.diagnostics
    counts = (
        diagnostics["completion_method"].value_counts().to_dict()
        if not diagnostics.empty and "completion_method" in diagnostics.columns
        else {}
    )
    requested = int(requested_count if requested_count is not None else len(diagnostics))
    completed = int(len(result.rows))
    return {
        "requested_count": requested,
        "completed_count": completed,
        "dropped_count": int(max(0, requested - completed)),
        "completion_method_counts": {str(k): int(v) for k, v in counts.items()},
        "sequences": _completion_sequence_summaries(result),
    }


def _completion_sequence_summaries(result: CompletionResult) -> dict[str, dict[str, Any]]:
    ids = set()
    if not result.diagnostics.empty and "sequence_id" in result.diagnostics.columns:
        ids.update(result.diagnostics["sequence_id"].dropna().astype(str))
    if not result.rows.empty and "sequence_id" in result.rows.columns:
        ids.update(result.rows["sequence_id"].dropna().astype(str))
    summaries = {}
    for sequence_id in sorted(ids):
        diag = _rows_for_sequence(result.diagnostics, sequence_id)
        rows = _rows_for_sequence(result.rows, sequence_id)
        counts = (
            diag["completion_method"].value_counts().to_dict()
            if not diag.empty and "completion_method" in diag.columns
            else {}
        )
        requested, completed = int(len(diag)), int(len(rows))
        summaries[sequence_id] = {
            "requested_count": requested,
            "completed_count": completed,
            "dropped_count": int(max(0, requested - completed)),
            "completion_method_counts": {str(k): int(v) for k, v in counts.items()},
            "completion_coverage_fraction": float(completed / requested) if requested else 0.0,
            "all_requested_timestamps_completed": bool(requested) and completed == requested,
        }
    return summaries


def _rows_for_sequence(frame: pd.DataFrame, sequence_id: str) -> pd.DataFrame:
    if frame.empty or "sequence_id" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["sequence_id"].astype(str) == sequence_id].copy()


def _normalize_max_interpolation_gap_s(value: float) -> float:
    try:
        gap = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_interpolation_gap_s must be a finite non-negative number") from exc
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError("max_interpolation_gap_s must be a finite non-negative number")
    return gap


def _complete_one(
    timestamp: float,
    times: np.ndarray,
    xyz: np.ndarray,
    scores: np.ndarray,
    max_gap: float,
    extrapolation: str,
):
    if len(times) == 1:
        t = float(times[0])
        if abs(t - timestamp) < 1e-9:
            return xyz[0], scores[0], "exact", t, t
        return (xyz[0], scores[0], "hold_single", t, t) if extrapolation == "hold" else None
    right = int(np.searchsorted(times, timestamp, side="left"))
    if right < len(times) and abs(float(times[right] - timestamp)) < 1e-9:
        return xyz[right], scores[right], "exact", float(times[right]), float(times[right])
    left = right - 1
    if 0 <= left and right < len(times):
        left_t, right_t = float(times[left]), float(times[right])
        gap = right_t - left_t
        if 0.0 < gap <= max_gap:
            alpha = (timestamp - left_t) / gap
            return (
                (1.0 - alpha) * xyz[left] + alpha * xyz[right],
                float((1.0 - alpha) * scores[left] + alpha * scores[right]),
                "interpolated",
                left_t,
                right_t,
            )
    if extrapolation != "hold":
        return None
    nearest = int(np.argmin(np.abs(times - timestamp)))
    method = "hold_before" if times[nearest] <= timestamp else "hold_after"
    return xyz[nearest], scores[nearest], method, float(times[nearest]), float(times[nearest])


def _mode_string(values: pd.Series) -> str:
    if values.empty:
        return "unknown"
    text = values.where(values.notna(), "").astype(str).str.strip()
    valid = text.loc[~text.str.lower().isin(_MISSING_TEXT_STRINGS)]
    if valid.empty:
        return "unknown"
    mode = valid.mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else str(valid.iloc[0])


def _diagnostic_row(
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
