"""Audit whether MMUAD mixture-MAP assigns weight to oracle candidates.

The reservoir-mixture runner can show that good candidates survived reservoir
construction and how well the final trajectory scored.  This diagnostic fills the
remaining gap: when the retained reservoir contains a good candidate, did the
mixture objective give that candidate weight, or did it put responsibility on a
worse branch/source?

The module is inference-safe when no truth is supplied elsewhere: truth is only
used here for local public-validation diagnostics and paper/debug artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import normalize_truth_columns

FRAME_CSV = "mmuad_mixture_assignment_oracle_frames.csv"
SUMMARY_CSV = "mmuad_mixture_assignment_oracle_summary.csv"
BY_SEQUENCE_CSV = "mmuad_mixture_assignment_oracle_by_sequence.csv"
SUMMARY_JSON = "mmuad_mixture_assignment_oracle_summary.json"
_TRUE_TEXT = {"1", "true", "t", "yes", "y"}
_FALSE_TEXT = {"", "0", "false", "f", "no", "n", "nan", "none", "null"}


def build_mixture_assignment_oracle_audit(
    assignments: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_truth_time_delta_s: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build frame, pooled, and per-sequence assignment-oracle audit tables."""

    rows = pd.DataFrame(assignments).copy()
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if rows.empty or truth_rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    _ensure_assignment_columns(rows)
    rows = rows.copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ["time_s", "x_m", "y_m", "z_m", "mixture_final_weight"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    if "candidate_rank" in rows.columns:
        rows["candidate_rank"] = pd.to_numeric(rows["candidate_rank"], errors="coerce")
    else:
        rows["candidate_rank"] = np.nan
    truth_rows["sequence_id"] = truth_rows["sequence_id"].astype(str)
    truth_by_sequence = {
        sequence_id: group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in truth_rows.groupby("sequence_id", sort=True)
    }
    frame_records: list[dict[str, Any]] = []
    for (sequence_id, time_s), group in rows.groupby(["sequence_id", "time_s"], sort=True):
        seq_truth = truth_by_sequence.get(str(sequence_id))
        if seq_truth is None or seq_truth.empty:
            continue
        truth_t = seq_truth["time_s"].to_numpy(float)
        nearest_idx = int(np.argmin(np.abs(truth_t - float(time_s))))
        truth_dt = float(time_s) - float(truth_t[nearest_idx])
        if abs(truth_dt) > float(max_truth_time_delta_s):
            continue
        truth_xyz = seq_truth.iloc[nearest_idx][["x_m", "y_m", "z_m"]].to_numpy(float)
        group = group.reset_index(drop=True).copy()
        finite_xyz = np.isfinite(group[["x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
        group = group.loc[finite_xyz].reset_index(drop=True)
        if group.empty:
            continue
        candidate_xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
        truth_distances = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
        weights = pd.to_numeric(group["mixture_final_weight"], errors="coerce").fillna(0.0)
        oracle_pos = int(np.argmin(truth_distances))
        dominant_pos = _dominant_position(group, weights)
        state_error = _state_error(group.iloc[dominant_pos], truth_xyz)
        oracle_weight_rank = _weight_rank(weights.to_numpy(float), oracle_pos)
        record = {
            "sequence_id": str(sequence_id),
            "time_s": float(time_s),
            "truth_time_delta_s": truth_dt,
            "candidate_count": int(len(group)),
            "state_error_3d_m": state_error,
            "oracle_candidate_error_3d_m": float(truth_distances[oracle_pos]),
            "dominant_candidate_error_3d_m": float(truth_distances[dominant_pos]),
            "dominant_minus_oracle_error_3d_m": float(
                truth_distances[dominant_pos] - truth_distances[oracle_pos]
            ),
            "state_minus_oracle_error_3d_m": (
                None if state_error is None else float(state_error - truth_distances[oracle_pos])
            ),
            "oracle_candidate_rank": _optional_int(group.iloc[oracle_pos].get("candidate_rank")),
            "oracle_weight_rank": int(oracle_weight_rank),
            "oracle_mixture_weight": float(weights.iloc[oracle_pos]),
            "dominant_mixture_weight": float(weights.iloc[dominant_pos]),
            "oracle_is_dominant": bool(oracle_pos == dominant_pos),
            "oracle_source": str(group.iloc[oracle_pos].get("source", "unknown")),
            "dominant_source": str(group.iloc[dominant_pos].get("source", "unknown")),
            "oracle_branch": str(group.iloc[oracle_pos].get("candidate_branch", "candidate")),
            "dominant_branch": str(group.iloc[dominant_pos].get("candidate_branch", "candidate")),
            "oracle_track_id": str(group.iloc[oracle_pos].get("track_id", "")),
            "dominant_track_id": str(group.iloc[dominant_pos].get("track_id", "")),
        }
        frame_records.append(record)
    frame_rows = pd.DataFrame.from_records(frame_records)
    if frame_rows.empty:
        empty = pd.DataFrame()
        return frame_rows, empty, empty
    pooled = pd.DataFrame.from_records([_summarize_assignment_oracle(frame_rows, "__pooled__")])
    by_sequence = pd.DataFrame.from_records(
        [
            _summarize_assignment_oracle(group, str(sequence_id))
            for sequence_id, group in frame_rows.groupby("sequence_id", sort=True)
        ]
    )
    return frame_rows, pooled, by_sequence


def write_mixture_assignment_oracle_audit_outputs(
    *,
    output_dir: Path,
    frame_rows: pd.DataFrame,
    summary: pd.DataFrame,
    by_sequence: pd.DataFrame,
) -> dict[str, Path]:
    """Write assignment-oracle audit artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "frame_csv": output / FRAME_CSV,
        "summary_csv": output / SUMMARY_CSV,
        "by_sequence_csv": output / BY_SEQUENCE_CSV,
        "summary_json": output / SUMMARY_JSON,
    }
    frame_rows.to_csv(paths["frame_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    by_sequence.to_csv(paths["by_sequence_csv"], index=False)
    payload = {
        "summary": summary.to_dict(orient="records"),
        "by_sequence": by_sequence.to_dict(orient="records"),
    }
    paths["summary_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.candidate_mixture_assignment_audit",
        description="audit MMUAD mixture assignment weights against oracle candidates",
    )
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    frame_rows, summary, by_sequence = build_mixture_assignment_oracle_audit(
        pd.read_csv(args.assignments_csv),
        pd.read_csv(args.truth_csv),
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
    )
    paths = write_mixture_assignment_oracle_audit_outputs(
        output_dir=args.output_dir,
        frame_rows=frame_rows,
        summary=summary,
        by_sequence=by_sequence,
    )
    print("mmuad_mixture_assignment_oracle_audit=ok")
    print(f"frame_count={len(frame_rows)}")
    if not summary.empty:
        print(f"oracle_dominant_fraction={summary.loc[0, 'oracle_dominant_fraction']}")
        print(f"state_mse_3d_m2={summary.loc[0, 'state_error_3d_m_mse']}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _ensure_assignment_columns(rows: pd.DataFrame) -> None:
    required = ["sequence_id", "time_s", "x_m", "y_m", "z_m", "mixture_final_weight"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"mixture assignment rows missing required columns: {missing}")


def _dominant_position(group: pd.DataFrame, weights: pd.Series) -> int:
    if "mixture_dominant" in group.columns:
        dominant = group["mixture_dominant"].map(_parse_bool_flag)
        if dominant.any():
            return int(np.flatnonzero(dominant.to_numpy(dtype=bool))[0])
    return int(np.argmax(weights.to_numpy(float)))


def _parse_bool_flag(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
        return bool(np.isfinite(number) and number != 0.0)
    text = str(value).strip().lower()
    if text in _TRUE_TEXT:
        return True
    if text in _FALSE_TEXT:
        return False
    return False


def _weight_rank(weights: np.ndarray, index: int) -> int:
    order = np.argsort(-np.asarray(weights, dtype=float), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1, dtype=int)
    return int(ranks[int(index)])


def _state_error(row: pd.Series, truth_xyz: np.ndarray) -> float | None:
    columns = ["state_x_m", "state_y_m", "state_z_m"]
    if not all(column in row.index for column in columns):
        return None
    state = pd.to_numeric(row[columns], errors="coerce").to_numpy(float)
    if not np.isfinite(state).all():
        return None
    return float(np.linalg.norm(state - truth_xyz))


def _summarize_assignment_oracle(frame_rows: pd.DataFrame, sequence_id: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sequence_id": sequence_id,
        "frame_count": int(len(frame_rows)),
        "candidate_count_mean": float(pd.to_numeric(frame_rows["candidate_count"]).mean()),
        "oracle_dominant_fraction": float(frame_rows["oracle_is_dominant"].mean()),
        "oracle_weight_mean": float(pd.to_numeric(frame_rows["oracle_mixture_weight"]).mean()),
        "oracle_weight_rank_mean": float(pd.to_numeric(frame_rows["oracle_weight_rank"]).mean()),
        "oracle_weight_rank_p95": float(
            pd.to_numeric(frame_rows["oracle_weight_rank"]).quantile(0.95)
        ),
    }
    for column in [
        "state_error_3d_m",
        "oracle_candidate_error_3d_m",
        "dominant_candidate_error_3d_m",
        "dominant_minus_oracle_error_3d_m",
        "state_minus_oracle_error_3d_m",
    ]:
        _add_error_stats(record, frame_rows, column)
    return _jsonable(record)


def _add_error_stats(record: dict[str, Any], rows: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty:
        record[f"{column}_mean"] = None
        record[f"{column}_mse"] = None
        record[f"{column}_p95"] = None
        record[f"{column}_max"] = None
        return
    array = values.to_numpy(float)
    record[f"{column}_mean"] = float(values.mean())
    record[f"{column}_mse"] = float(np.mean(array**2))
    record[f"{column}_p95"] = float(values.quantile(0.95))
    record[f"{column}_max"] = float(values.max())


def _optional_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
