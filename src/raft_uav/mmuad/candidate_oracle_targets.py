"""Build MMUAD candidate-level oracle targets for re-ranking experiments.

The current MMUAD pose gap is dominated by candidate assignment: the full
candidate pool often contains a good target, but a single score ordering can
bury it before robust mixture smoothing uses it.  This module converts any
candidate/reservoir/assignment CSV plus local truth into supervised candidate
rows for train-only ranker, uncertainty, and soft-mixture-prior experiments.

Truth is required and the output is diagnostic/training-only; this command is
not part of hidden-test inference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns

TARGETS_CSV = "mmuad_candidate_oracle_targets.csv"
FRAME_SUMMARY_CSV = "mmuad_candidate_oracle_target_frame_summary.csv"
SUMMARY_JSON = "mmuad_candidate_oracle_target_summary.json"


@dataclass(frozen=True)
class CandidateOracleTargetConfig:
    """Configuration for candidate-level oracle target export."""

    max_truth_time_delta_s: float = 0.5
    score_column: str = "candidate_reservoir_score"
    fallback_score_columns: tuple[str, ...] = (
        "candidate_reservoir_grid_score",
        "ranker_score",
        "confidence",
    )
    soft_tau_m: tuple[float, ...] = (2.0, 5.0, 10.0)
    good_thresholds_m: tuple[float, ...] = (2.0, 5.0, 10.0)


def build_candidate_oracle_targets(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    config: CandidateOracleTargetConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return per-candidate oracle labels, frame summaries, and a JSON summary."""

    config = config or CandidateOracleTargetConfig()
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if rows.empty or truth_rows.empty:
        empty = pd.DataFrame()
        return empty, empty, _summary_payload(empty, empty, config=config)
    _require_columns(rows, ["sequence_id", "time_s", "x_m", "y_m", "z_m"])
    rows = rows.copy().reset_index(drop=True)
    rows["_candidate_input_row"] = np.arange(len(rows), dtype=int)
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy()
    if rows.empty:
        empty = pd.DataFrame()
        return empty, empty, _summary_payload(empty, empty, config=config)
    rows["candidate_oracle_score"] = _candidate_score(rows, config=config)
    truth_by_sequence = {
        str(sequence_id): group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in truth_rows.groupby("sequence_id", sort=False)
    }

    target_parts: list[pd.DataFrame] = []
    frame_records: list[dict[str, Any]] = []
    for (sequence_id, time_s), group in rows.groupby(["sequence_id", "time_s"], sort=True):
        sequence_truth = truth_by_sequence.get(str(sequence_id))
        if sequence_truth is None or sequence_truth.empty:
            continue
        truth_row, truth_dt = _nearest_truth_row(sequence_truth, float(time_s))
        if truth_row is None or truth_dt > float(config.max_truth_time_delta_s):
            continue
        target, frame_record = _frame_targets(
            group,
            truth_row,
            truth_dt=truth_dt,
            config=config,
        )
        target_parts.append(target)
        frame_records.append(frame_record)
    target_rows = pd.concat(target_parts, ignore_index=True) if target_parts else pd.DataFrame()
    frame_summary = pd.DataFrame.from_records(frame_records)
    summary = _summary_payload(target_rows, frame_summary, config=config)
    return target_rows, frame_summary, summary


def write_candidate_oracle_target_outputs(
    *,
    target_rows: pd.DataFrame,
    frame_summary: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write candidate target rows and summaries."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "targets_csv": output / TARGETS_CSV,
        "frame_summary_csv": output / FRAME_SUMMARY_CSV,
        "summary_json": output / SUMMARY_JSON,
    }
    target_rows.to_csv(paths["targets_csv"], index=False)
    frame_summary.to_csv(paths["frame_summary_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-oracle-targets",
        description="build train-only MMUAD candidate oracle targets for ranker experiments",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--score-column", default="candidate_reservoir_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--soft-tau-m", type=float, action="append", default=[])
    parser.add_argument("--good-threshold-m", type=float, action="append", default=[])
    args = parser.parse_args(argv)

    config = CandidateOracleTargetConfig(
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
        score_column=str(args.score_column),
        fallback_score_columns=tuple(args.fallback_score_column)
        or ("candidate_reservoir_grid_score", "ranker_score", "confidence"),
        soft_tau_m=tuple(args.soft_tau_m) or (2.0, 5.0, 10.0),
        good_thresholds_m=tuple(args.good_threshold_m) or (2.0, 5.0, 10.0),
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    truth = load_evaluation_truth_file(args.truth_csv).rows
    target_rows, frame_summary, summary = build_candidate_oracle_targets(
        candidates,
        truth,
        config=config,
    )
    paths = write_candidate_oracle_target_outputs(
        target_rows=target_rows,
        frame_summary=frame_summary,
        summary=summary,
        output_dir=args.output_dir,
    )
    print("mmuad_candidate_oracle_targets=ok")
    print(f"candidate_rows={len(target_rows)}")
    print(f"frame_count={len(frame_summary)}")
    pooled = summary.get("pooled", {})
    if pooled.get("oracle_mse_3d_m2") is not None:
        print(f"oracle_mse_3d_m2={pooled['oracle_mse_3d_m2']}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _frame_targets(
    group: pd.DataFrame,
    truth_row: pd.Series,
    *,
    truth_dt: float,
    config: CandidateOracleTargetConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    group = group.copy().reset_index(drop=True)
    truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
    xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
    errors = np.linalg.norm(xyz - truth_xyz.reshape(1, 3), axis=1)
    oracle_index = int(np.argmin(errors))
    score = pd.to_numeric(group["candidate_oracle_score"], errors="coerce").fillna(0.0)
    score_rank_order = np.argsort(-score.to_numpy(float), kind="stable")
    score_rank = np.empty(len(group), dtype=int)
    score_rank[score_rank_order] = np.arange(1, len(group) + 1, dtype=int)
    group["truth_time_s"] = float(truth_row["time_s"])
    group["truth_time_delta_s"] = float(truth_dt)
    group["truth_x_m"] = float(truth_xyz[0])
    group["truth_y_m"] = float(truth_xyz[1])
    group["truth_z_m"] = float(truth_xyz[2])
    group["candidate_error_3d_m"] = errors
    group["candidate_error_3d_m2"] = errors**2
    group["candidate_score_rank"] = score_rank
    group["candidate_is_oracle"] = False
    group.loc[oracle_index, "candidate_is_oracle"] = True
    group["candidate_regret_to_oracle_m"] = errors - float(errors[oracle_index])
    for threshold in config.good_thresholds_m:
        label = _threshold_label(threshold)
        group[f"candidate_good_le_{label}_m"] = errors <= float(threshold)
    for tau in config.soft_tau_m:
        tau = float(tau)
        label = _threshold_label(tau)
        weights = _soft_oracle_weights(errors, tau=tau)
        group[f"soft_oracle_weight_tau_{label}_m"] = weights

    frame_record: dict[str, Any] = {
        "sequence_id": str(group["sequence_id"].iloc[0]),
        "time_s": float(group["time_s"].iloc[0]),
        "truth_time_s": float(truth_row["time_s"]),
        "truth_time_delta_s": float(truth_dt),
        "candidate_count": int(len(group)),
        "oracle_error_3d_m": float(errors[oracle_index]),
        "score_top1_error_3d_m": float(errors[score_rank_order[0]]),
        "score_top1_regret_m": float(errors[score_rank_order[0]] - errors[oracle_index]),
        "oracle_score_rank": int(score_rank[oracle_index]),
        "oracle_source": _safe_str(group.iloc[oracle_index].get("source")),
        "oracle_candidate_branch": _safe_str(group.iloc[oracle_index].get("candidate_branch")),
    }
    for top_k in (1, 3, 5, 10, 20):
        bounded = min(top_k, len(group))
        frame_record[f"score_top{top_k}_oracle_error_3d_m"] = float(
            np.min(errors[score_rank_order[:bounded]])
        )
    return group, frame_record


def _summary_payload(
    target_rows: pd.DataFrame,
    frame_summary: pd.DataFrame,
    *,
    config: CandidateOracleTargetConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "raft-uav-mmuad-candidate-oracle-targets-v1",
        "config": asdict(config),
        "candidate_rows": int(len(target_rows)),
        "frame_count": int(len(frame_summary)),
    }
    if frame_summary.empty:
        payload["pooled"] = {}
        payload["by_sequence"] = []
        return _jsonable(payload)
    pooled = _summary_record(frame_summary, sequence_id="__pooled__")
    by_sequence = [
        _summary_record(group, sequence_id=str(sequence_id))
        for sequence_id, group in frame_summary.groupby("sequence_id", sort=True)
    ]
    payload["pooled"] = pooled
    payload["by_sequence"] = by_sequence
    return _jsonable(payload)


def _summary_record(frame_summary: pd.DataFrame, *, sequence_id: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sequence_id": sequence_id,
        "frame_count": int(len(frame_summary)),
        "candidate_count_mean": float(pd.to_numeric(frame_summary["candidate_count"]).mean()),
    }
    for column in frame_summary.columns:
        if not column.endswith("_3d_m"):
            continue
        values = pd.to_numeric(frame_summary[column], errors="coerce").dropna().to_numpy(float)
        if len(values) == 0:
            continue
        prefix = column[: -len("_3d_m")]
        mse = float(np.mean(values**2))
        record[f"{prefix}_mse_3d_m2"] = mse
        record[f"{prefix}_rmse_3d_m"] = float(np.sqrt(mse))
        record[f"{prefix}_p95_3d_m"] = float(np.quantile(values, 0.95))
        record[f"{prefix}_max_3d_m"] = float(np.max(values))
    if "oracle_score_rank" in frame_summary.columns:
        ranks = pd.to_numeric(frame_summary["oracle_score_rank"], errors="coerce").dropna()
        if not ranks.empty:
            record["oracle_score_rank_mean"] = float(ranks.mean())
            record["oracle_score_rank_p95"] = float(ranks.quantile(0.95))
            for top_k in (1, 3, 5, 10, 20):
                record[f"oracle_in_score_top{top_k}_fraction"] = float((ranks <= top_k).mean())
    return _jsonable(record)


def _candidate_score(rows: pd.DataFrame, *, config: CandidateOracleTargetConfig) -> pd.Series:
    columns = (config.score_column, *config.fallback_score_columns)
    result = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        result = result.where(result.notna(), values)
    return result.fillna(0.0).astype(float)


def _soft_oracle_weights(errors: np.ndarray, *, tau: float) -> np.ndarray:
    tau = max(float(tau), 1.0e-6)
    values = -0.5 * (np.asarray(errors, dtype=float) / tau) ** 2
    values = values - float(np.max(values))
    weights = np.exp(np.clip(values, -700.0, 0.0))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(errors), 1.0 / max(len(errors), 1), dtype=float)
    return weights / total


def _nearest_truth_row(truth_rows: pd.DataFrame, time_s: float) -> tuple[pd.Series | None, float]:
    if truth_rows.empty:
        return None, float("inf")
    deltas = np.abs(pd.to_numeric(truth_rows["time_s"], errors="coerce").to_numpy(float) - time_s)
    if not np.isfinite(deltas).any():
        return None, float("inf")
    index = int(np.nanargmin(deltas))
    return truth_rows.iloc[index], float(deltas[index])


def _require_columns(rows: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(f"candidate rows missing required columns: {missing}")


def _threshold_label(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return text


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
