"""Diagnose candidate-mixture assignment gaps against local MMUAD truth.

The branch-preserving reservoir and robust mixture-MAP commands can now keep many
candidate hypotheses alive.  This diagnostic answers the next question: whether
poor pose rows are caused by a missing good candidate, a good candidate buried in
low mixture responsibility, a wrong dominant assignment, or the smooth trajectory
remaining far from an otherwise good assignment.

The module is intended for validation/train diagnostics only.  It requires truth
and is not part of the hidden-test inference path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.schema import normalize_truth_columns

FRAME_CSV = "mmuad_candidate_assignment_diagnostics.csv"
SUMMARY_CSV = "mmuad_candidate_assignment_summary.csv"
SUMMARY_JSON = "mmuad_candidate_assignment_summary.json"


@dataclass(frozen=True)
class CandidateAssignmentDiagnosticsConfig:
    """Configuration for local candidate-assignment diagnostics."""

    max_truth_time_delta_s: float = 0.5
    good_candidate_threshold_m: float = 5.0
    regret_threshold_m: float = 2.0
    top_k: int = 3


def build_candidate_assignment_diagnostics(
    assignments: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    config: CandidateAssignmentDiagnosticsConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return frame-level and summary diagnostics for mixture assignments.

    ``assignments`` should be the CSV emitted by
    ``raft-uav-mmuad-candidate-mixture-map`` or
    ``raft-uav-mmuad-reservoir-mixture-map``.  The diagnostic matches each
    assignment frame to nearest truth in the same sequence and compares the
    oracle candidate with the dominant/weighted/state trajectory outputs.
    """

    config = config or CandidateAssignmentDiagnosticsConfig()
    rows = pd.DataFrame(assignments).copy()
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if rows.empty or truth_rows.empty:
        empty = pd.DataFrame()
        return empty, empty
    _require_columns(rows, ["sequence_id", "time_s", "x_m", "y_m", "z_m"])
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.loc[np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    if rows.empty:
        empty = pd.DataFrame()
        return empty, empty

    truth_by_sequence = {
        str(sequence_id): group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in truth_rows.groupby("sequence_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    for (sequence_id, time_s), group in rows.groupby(["sequence_id", "time_s"], sort=True):
        sequence_truth = truth_by_sequence.get(str(sequence_id))
        if sequence_truth is None or sequence_truth.empty:
            continue
        truth_row, truth_dt = _nearest_truth_row(sequence_truth, float(time_s))
        if truth_row is None or truth_dt > float(config.max_truth_time_delta_s):
            continue
        records.append(_frame_record(group, truth_row, truth_dt=truth_dt, config=config))
    frame_rows = pd.DataFrame.from_records(records)
    summary = build_candidate_assignment_summary(frame_rows)
    return frame_rows, summary


def build_candidate_assignment_summary(frame_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate-assignment diagnostics by sequence and failure mode."""

    rows = pd.DataFrame(frame_rows).copy()
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    records.append(_summary_record(rows, sequence_id="__pooled__", failure_mode="__all__"))
    for mode, group in rows.groupby("assignment_failure_mode", sort=True):
        records.append(_summary_record(group, sequence_id="__pooled__", failure_mode=str(mode)))
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        records.append(_summary_record(group, sequence_id=str(sequence_id), failure_mode="__all__"))
        for mode, mode_group in group.groupby("assignment_failure_mode", sort=True):
            records.append(
                _summary_record(mode_group, sequence_id=str(sequence_id), failure_mode=str(mode))
            )
    return pd.DataFrame.from_records(records)


def write_candidate_assignment_diagnostics(
    *,
    frame_rows: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    config: CandidateAssignmentDiagnosticsConfig,
) -> dict[str, Path]:
    """Write frame and summary diagnostics."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "frame_csv": output / FRAME_CSV,
        "summary_csv": output / SUMMARY_CSV,
        "summary_json": output / SUMMARY_JSON,
    }
    frame_rows.to_csv(paths["frame_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    payload = {
        "schema": "raft-uav-mmuad-candidate-assignment-diagnostics-v1",
        "config": asdict(config),
        "frame_count": int(len(frame_rows)),
        "summary": summary.to_dict(orient="records"),
    }
    paths["summary_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-assignment-diagnostics",
        description="diagnose MMUAD candidate-mixture assignment gaps against local truth",
    )
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--good-candidate-threshold-m", type=float, default=5.0)
    parser.add_argument("--regret-threshold-m", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    config = CandidateAssignmentDiagnosticsConfig(
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
        good_candidate_threshold_m=float(args.good_candidate_threshold_m),
        regret_threshold_m=float(args.regret_threshold_m),
        top_k=int(args.top_k),
    )
    assignments = pd.read_csv(args.assignments_csv)
    truth = load_evaluation_truth_file(args.truth_csv).rows
    frame_rows, summary = build_candidate_assignment_diagnostics(
        assignments,
        truth,
        config=config,
    )
    paths = write_candidate_assignment_diagnostics(
        frame_rows=frame_rows,
        summary=summary,
        output_dir=args.output_dir,
        config=config,
    )
    print("mmuad_candidate_assignment_diagnostics=ok")
    print(f"frame_count={len(frame_rows)}")
    if not summary.empty:
        pooled = summary.loc[
            (summary["sequence_id"] == "__pooled__")
            & (summary["assignment_failure_mode"] == "__all__")
        ]
        if not pooled.empty:
            print(f"state_error_3d_m_mse={pooled.iloc[0]['state_error_3d_m_mse']}")
            print(f"oracle_error_3d_m_mse={pooled.iloc[0]['oracle_error_3d_m_mse']}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _frame_record(
    group: pd.DataFrame,
    truth_row: pd.Series,
    *,
    truth_dt: float,
    config: CandidateAssignmentDiagnosticsConfig,
) -> dict[str, Any]:
    group = group.copy().reset_index(drop=True)
    xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
    truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
    errors = np.linalg.norm(xyz - truth_xyz.reshape(1, 3), axis=1)
    oracle_index = int(np.argmin(errors))
    weights = _assignment_weights(group)
    weighted_xyz = np.sum(weights[:, None] * xyz, axis=0)
    dominant_index = _dominant_index(group, weights)
    state_xyz = _state_xyz(group, weighted_xyz)
    state_error = float(np.linalg.norm(state_xyz - truth_xyz))
    weighted_error = float(np.linalg.norm(weighted_xyz - truth_xyz))
    oracle_error = float(errors[oracle_index])
    dominant_error = float(errors[dominant_index])
    weight_rank_order = np.argsort(-weights, kind="stable")
    weight_rank = {int(index): int(rank + 1) for rank, index in enumerate(weight_rank_order)}
    oracle_weight_rank = int(weight_rank[oracle_index])
    dominant_candidate = group.iloc[dominant_index]
    oracle_candidate = group.iloc[oracle_index]
    failure_mode = _assignment_failure_mode(
        oracle_error_m=oracle_error,
        dominant_error_m=dominant_error,
        state_error_m=state_error,
        oracle_weight_rank=oracle_weight_rank,
        config=config,
    )
    return {
        "sequence_id": str(group["sequence_id"].iloc[0]),
        "time_s": float(group["time_s"].iloc[0]),
        "truth_time_s": float(truth_row["time_s"]),
        "truth_time_delta_s": float(truth_dt),
        "candidate_count": int(len(group)),
        "assignment_failure_mode": failure_mode,
        "oracle_error_3d_m": oracle_error,
        "dominant_error_3d_m": dominant_error,
        "weighted_error_3d_m": weighted_error,
        "state_error_3d_m": state_error,
        "dominant_regret_m": dominant_error - oracle_error,
        "weighted_regret_m": weighted_error - oracle_error,
        "state_regret_m": state_error - oracle_error,
        "oracle_weight_rank": oracle_weight_rank,
        "oracle_candidate_rank": _safe_int(oracle_candidate.get("candidate_rank")),
        "oracle_mixture_weight": float(weights[oracle_index]),
        "oracle_source": _safe_str(oracle_candidate.get("source")),
        "oracle_track_id": _safe_str(oracle_candidate.get("track_id")),
        "oracle_candidate_branch": _safe_str(oracle_candidate.get("candidate_branch")),
        "dominant_weight_rank": int(weight_rank[dominant_index]),
        "dominant_candidate_rank": _safe_int(dominant_candidate.get("candidate_rank")),
        "dominant_mixture_weight": float(weights[dominant_index]),
        "dominant_source": _safe_str(dominant_candidate.get("source")),
        "dominant_track_id": _safe_str(dominant_candidate.get("track_id")),
        "dominant_candidate_branch": _safe_str(dominant_candidate.get("candidate_branch")),
        "dominant_is_oracle": bool(dominant_index == oracle_index),
        "oracle_in_topk_by_weight": bool(oracle_weight_rank <= int(config.top_k)),
        "max_mixture_weight": float(np.max(weights)),
        "assignment_entropy": _entropy(weights),
        "state_x_m": float(state_xyz[0]),
        "state_y_m": float(state_xyz[1]),
        "state_z_m": float(state_xyz[2]),
        "weighted_x_m": float(weighted_xyz[0]),
        "weighted_y_m": float(weighted_xyz[1]),
        "weighted_z_m": float(weighted_xyz[2]),
        "truth_x_m": float(truth_xyz[0]),
        "truth_y_m": float(truth_xyz[1]),
        "truth_z_m": float(truth_xyz[2]),
    }


def _assignment_failure_mode(
    *,
    oracle_error_m: float,
    dominant_error_m: float,
    state_error_m: float,
    oracle_weight_rank: int,
    config: CandidateAssignmentDiagnosticsConfig,
) -> str:
    if oracle_error_m > float(config.good_candidate_threshold_m):
        return "missing_good_candidate_in_assignments"
    if oracle_weight_rank > int(config.top_k):
        return "good_candidate_buried"
    regret = float(config.regret_threshold_m)
    if dominant_error_m - oracle_error_m > regret:
        return "wrong_dominant_assignment"
    if state_error_m - oracle_error_m > regret:
        return "smoothing_assignment_gap"
    return "covered"


def _assignment_weights(group: pd.DataFrame) -> np.ndarray:
    if "mixture_final_weight" in group.columns:
        weights = pd.to_numeric(group["mixture_final_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    elif "mixture_dominant" in group.columns:
        weights = np.asarray([bool(value) for value in group["mixture_dominant"]], dtype=float)
    else:
        weights = np.ones(len(group), dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= 1.0e-12:
        return np.ones(len(group), dtype=float) / max(float(len(group)), 1.0)
    return weights / total


def _dominant_index(group: pd.DataFrame, weights: np.ndarray) -> int:
    if "mixture_dominant" in group.columns:
        dominant_mask = group["mixture_dominant"].astype(str).str.lower().isin({"true", "1", "yes"})
        if dominant_mask.any():
            return int(np.flatnonzero(dominant_mask.to_numpy())[0])
    return int(np.argmax(weights))


def _state_xyz(group: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    columns = ["state_x_m", "state_y_m", "state_z_m"]
    if all(column in group.columns for column in columns):
        state = group.loc[group.index[0], columns].to_numpy(dtype=float)
        if np.isfinite(state).all():
            return state
    return np.asarray(fallback, dtype=float)


def _nearest_truth_row(truth_rows: pd.DataFrame, time_s: float) -> tuple[pd.Series | None, float]:
    times = truth_rows["time_s"].to_numpy(float)
    if len(times) == 0:
        return None, float("inf")
    index = int(np.searchsorted(times, time_s, side="left"))
    candidates = []
    if index < len(times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    nearest = min(candidates, key=lambda item: abs(float(times[item]) - time_s))
    return truth_rows.iloc[nearest], abs(float(times[nearest]) - time_s)


def _summary_record(rows: pd.DataFrame, *, sequence_id: str, failure_mode: str) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "assignment_failure_mode": failure_mode,
        "frame_count": int(len(rows)),
        "oracle_error_3d_m_mse": _mse(rows["oracle_error_3d_m"]),
        "state_error_3d_m_mse": _mse(rows["state_error_3d_m"]),
        "dominant_error_3d_m_mse": _mse(rows["dominant_error_3d_m"]),
        "weighted_error_3d_m_mse": _mse(rows["weighted_error_3d_m"]),
        "oracle_error_3d_m_mean": _mean(rows["oracle_error_3d_m"]),
        "state_error_3d_m_mean": _mean(rows["state_error_3d_m"]),
        "state_error_3d_m_p95": _quantile(rows["state_error_3d_m"], 0.95),
        "state_error_3d_m_max": _max(rows["state_error_3d_m"]),
        "state_regret_m_mean": _mean(rows["state_regret_m"]),
        "state_regret_m_p95": _quantile(rows["state_regret_m"], 0.95),
        "dominant_regret_m_mean": _mean(rows["dominant_regret_m"]),
        "weighted_regret_m_mean": _mean(rows["weighted_regret_m"]),
        "oracle_in_topk_by_weight_rate": _mean(rows["oracle_in_topk_by_weight"].astype(float)),
        "dominant_matches_oracle_rate": _mean(rows["dominant_is_oracle"].astype(float)),
        "assignment_entropy_mean": _mean(rows["assignment_entropy"]),
        "dominant_oracle_branch": _mode(rows.get("oracle_candidate_branch")),
        "dominant_oracle_source": _mode(rows.get("oracle_source")),
    }


def _require_columns(rows: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(f"assignment rows missing required columns: {missing}")


def _mse(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(np.mean(values.to_numpy(float) ** 2))


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _quantile(values: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(quantile)) if not values.empty else float("nan")


def _max(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.max()) if not values.empty else float("nan")


def _mode(values: pd.Series | None) -> str:
    if values is None:
        return ""
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    return str(cleaned.value_counts().index[0])


def _entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[weights > 0.0]
    if len(weights) == 0:
        return 0.0
    return float(-np.sum(weights * np.log(weights)))


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


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
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
