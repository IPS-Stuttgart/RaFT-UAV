"""Risk-aware train-CV selection for MMUAD candidate reservoirs.

Mean held-out oracle recall can hide a reservoir configuration that performs well
on most training sequences but catastrophically prunes the useful physical mode
on one difficult sequence.  This module selects one inference-time reservoir
configuration with a convex mean/tail objective computed on training folds only:

    risk_score = (1 - alpha) * mean_metric + alpha * tail_metric

``alpha=0`` exactly recovers mean-CV selection.  ``alpha>0`` trades a small mean
penalty for protection against difficult sequences.  Validation/test inference
uses only the frozen branch/source offsets and never reads truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs
from raft_uav.mmuad.candidate_reservoir_grid import run_candidate_reservoir_offset_grid
from raft_uav.mmuad.schema import normalize_truth_columns

_DEFAULT_TOP_K = (1, 3, 5, 10, 20)
_DEFAULT_SELECTION_METRIC = "oracle_top5_3d_m_mse"


def build_risk_aggregate_summary(
    fold_summary: pd.DataFrame,
    *,
    selection_metric: str,
    risk_aversion: float = 0.5,
    tail_quantile: float = 1.0,
) -> pd.DataFrame:
    """Aggregate held-out grid rows with a mean/tail risk objective.

    The selection metric is assumed to be lower-is-better, as for the MMUAD
    oracle MSE metrics used by the candidate-reservoir grid.
    """

    _validate_risk_parameters(risk_aversion, tail_quantile)
    rows = pd.DataFrame(fold_summary).copy()
    if rows.empty or selection_metric not in rows.columns:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for grid_label, group in rows.groupby("grid_label", sort=False):
        values = pd.to_numeric(group[selection_metric], errors="coerce").dropna()
        if values.empty:
            continue
        first = group.iloc[0]
        mean_value = float(values.mean())
        tail_value = float(values.quantile(float(tail_quantile), interpolation="linear"))
        risk_score = (1.0 - float(risk_aversion)) * mean_value + (
            float(risk_aversion) * tail_value
        )
        records.append(
            {
                "grid_label": str(grid_label),
                "branch_score_offsets_json": first.get(
                    "branch_score_offsets_json",
                    "{}",
                ),
                "source_score_offsets_json": first.get(
                    "source_score_offsets_json",
                    "{}",
                ),
                "fold_count": int(len(values)),
                f"{selection_metric}_mean": mean_value,
                f"{selection_metric}_std": float(values.std(ddof=0)),
                f"{selection_metric}_min": float(values.min()),
                f"{selection_metric}_max": float(values.max()),
                f"{selection_metric}_tail_q{_quantile_label(tail_quantile)}": tail_value,
                f"{selection_metric}_risk_score": float(risk_score),
                "risk_aversion": float(risk_aversion),
                "tail_quantile": float(tail_quantile),
            }
        )
    out = pd.DataFrame.from_records(records)
    risk_column = f"{selection_metric}_risk_score"
    mean_column = f"{selection_metric}_mean"
    if not out.empty:
        out = out.sort_values(
            [risk_column, mean_column, "grid_label"],
            na_position="last",
        ).reset_index(drop=True)
    return out


def select_candidate_reservoir_offsets_by_risk_cv(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    branch_offset_grid: Sequence[str] = (),
    source_offset_grid: Sequence[str] = (),
    score_column: str = "ranker_score",
    fallback_score_column: str = "confidence",
    global_top_n: int = 20,
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    max_candidates_per_frame: int = 40,
    score_floor_quantile: float | None = None,
    top_k_values: Sequence[int] = _DEFAULT_TOP_K,
    max_truth_time_delta_s: float = 0.5,
    selection_metric: str = _DEFAULT_SELECTION_METRIC,
    risk_aversion: float = 0.5,
    tail_quantile: float = 1.0,
    write_best_reservoir: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Select one fixed reservoir configuration with held-out tail protection."""

    _validate_risk_parameters(risk_aversion, tail_quantile)
    candidate_rows, truth_rows, sequences = _validated_inputs(candidates, truth)
    fold_frames: list[pd.DataFrame] = []
    for holdout in sequences:
        holdout_candidates = candidate_rows.loc[
            candidate_rows["sequence_id"].astype(str) == holdout
        ]
        holdout_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) == holdout]
        if holdout_candidates.empty or holdout_truth.empty:
            continue
        holdout_summary, _ = run_candidate_reservoir_offset_grid(
            holdout_candidates,
            truth=holdout_truth,
            branch_offset_grid=branch_offset_grid,
            source_offset_grid=source_offset_grid,
            score_column=score_column,
            fallback_score_column=fallback_score_column,
            global_top_n=global_top_n,
            per_source_top_n=per_source_top_n,
            per_branch_top_n=per_branch_top_n,
            max_candidates_per_frame=max_candidates_per_frame,
            score_floor_quantile=score_floor_quantile,
            top_k_values=top_k_values,
            max_truth_time_delta_s=max_truth_time_delta_s,
            selection_metric=selection_metric,
        )
        if holdout_summary.empty:
            continue
        holdout_summary = holdout_summary.copy()
        holdout_summary.insert(0, "holdout_sequence_id", holdout)
        fold_frames.append(holdout_summary)
    if not fold_frames:
        raise ValueError("risk-aware CV selection did not produce any fold summaries")
    fold_summary = pd.concat(fold_frames, ignore_index=True)
    aggregate_summary = build_risk_aggregate_summary(
        fold_summary,
        selection_metric=selection_metric,
        risk_aversion=risk_aversion,
        tail_quantile=tail_quantile,
    )
    if aggregate_summary.empty:
        raise ValueError(f"risk-aware CV summary missing metric {selection_metric!r}")

    selected = aggregate_summary.iloc[0]
    selected_branch_offsets = _json_dict(selected.get("branch_score_offsets_json", "{}"))
    selected_source_offsets = _json_dict(selected.get("source_score_offsets_json", "{}"))
    final_summary, best_reservoir = run_candidate_reservoir_offset_grid(
        candidate_rows,
        truth=truth_rows,
        branch_offset_grid=_offset_dict_to_specs(selected_branch_offsets),
        source_offset_grid=_offset_dict_to_specs(selected_source_offsets),
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        global_top_n=global_top_n,
        per_source_top_n=per_source_top_n,
        per_branch_top_n=per_branch_top_n,
        max_candidates_per_frame=max_candidates_per_frame,
        score_floor_quantile=score_floor_quantile,
        top_k_values=top_k_values,
        max_truth_time_delta_s=max_truth_time_delta_s,
        selection_metric=selection_metric,
        write_best_reservoir=write_best_reservoir,
    )
    if final_summary.empty:
        raise ValueError("selected risk-aware config did not produce a train summary")

    tail_column = f"{selection_metric}_tail_q{_quantile_label(tail_quantile)}"
    risk_column = f"{selection_metric}_risk_score"
    mean_column = f"{selection_metric}_mean"
    std_column = f"{selection_metric}_std"
    max_column = f"{selection_metric}_max"
    selected_config: dict[str, Any] = {
        "schema_version": 1,
        "selection_protocol": (
            "leave-one-sequence-out-cv-risk-aggregate__final-apply-selected-offsets"
        ),
        "sequence_count": int(len(sequences)),
        "selection_metric": selection_metric,
        "selected_grid_label": str(selected.get("grid_label", "")),
        "selected_metric_value": float(selected.get(mean_column, float("nan"))),
        "selected_metric_std": float(selected.get(std_column, float("nan"))),
        "selected_metric_max": float(selected.get(max_column, float("nan"))),
        "selected_tail_metric_value": float(selected.get(tail_column, float("nan"))),
        "selected_risk_score": float(selected.get(risk_column, float("nan"))),
        "selected_metric_fold_count": int(selected.get("fold_count", 0)),
        "risk_aversion": float(risk_aversion),
        "tail_quantile": float(tail_quantile),
        "branch_score_offsets": selected_branch_offsets,
        "source_score_offsets": selected_source_offsets,
        "score_column": score_column,
        "fallback_score_column": fallback_score_column,
        "global_top_n": int(global_top_n),
        "per_source_top_n": int(per_source_top_n),
        "per_branch_top_n": int(per_branch_top_n),
        "max_candidates_per_frame": int(max_candidates_per_frame),
        "score_floor_quantile": score_floor_quantile,
        "top_k_values": [int(value) for value in top_k_values],
        "max_truth_time_delta_s": float(max_truth_time_delta_s),
        "all_train_selected_metric_value": float(
            final_summary.iloc[0].get(selection_metric, float("nan"))
        ),
    }
    return (
        _jsonable(selected_config),
        fold_summary,
        aggregate_summary,
        best_reservoir if write_best_reservoir else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmuad-candidate-reservoir-risk-cv",
        description=(
            "select MMUAD candidate-reservoir offsets with train-only mean/tail CV risk"
        ),
    )
    parser.add_argument("--candidate", action="append", default=[], help="BRANCH=path")
    parser.add_argument("--candidate-csv", action="append", default=[], help="alias")
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch-score-offset-grid", action="append", default=[])
    parser.add_argument("--source-score-offset-grid", action="append", default=[])
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--top-k", action="append", type=int, default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--selection-metric", default=_DEFAULT_SELECTION_METRIC)
    parser.add_argument("--risk-aversion", type=float, default=0.5)
    parser.add_argument("--tail-quantile", type=float, default=1.0)
    parser.add_argument("--write-best-reservoir", action="store_true")
    args = parser.parse_args(argv)

    candidates = load_candidate_inputs([*args.candidate, *args.candidate_csv])
    truth = pd.read_csv(args.truth_csv)
    selected_config, folds, aggregate, best_reservoir = (
        select_candidate_reservoir_offsets_by_risk_cv(
            candidates,
            truth,
            branch_offset_grid=args.branch_score_offset_grid,
            source_offset_grid=args.source_score_offset_grid,
            score_column=args.score_column,
            fallback_score_column=args.fallback_score_column,
            global_top_n=args.global_top_n,
            per_source_top_n=args.per_source_top_n,
            per_branch_top_n=args.per_branch_top_n,
            max_candidates_per_frame=args.max_candidates_per_frame,
            score_floor_quantile=args.score_floor_quantile,
            top_k_values=tuple(args.top_k or _DEFAULT_TOP_K),
            max_truth_time_delta_s=args.max_truth_time_delta_s,
            selection_metric=args.selection_metric,
            risk_aversion=args.risk_aversion,
            tail_quantile=args.tail_quantile,
            write_best_reservoir=args.write_best_reservoir,
        )
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config_json = output / "mmuad_candidate_reservoir_risk_cv_selected_config.json"
    folds_csv = output / "mmuad_candidate_reservoir_risk_cv_folds.csv"
    aggregate_csv = output / "mmuad_candidate_reservoir_risk_cv_aggregate.csv"
    config_json.write_text(json.dumps(selected_config, indent=2), encoding="utf-8")
    folds.to_csv(folds_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)
    if best_reservoir is not None:
        best_reservoir.to_csv(
            output / "mmuad_candidate_reservoir_risk_cv_selected.csv",
            index=False,
        )
    print("mmuad_candidate_reservoir_risk_cv=ok")
    print(f"selected_grid_label={selected_config['selected_grid_label']}")
    print(f"selected_risk_score={selected_config['selected_risk_score']}")
    print(f"selected_config_json={config_json}")
    print(f"folds_csv={folds_csv}")
    print(f"aggregate_csv={aggregate_csv}")
    return 0


def _validated_inputs(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    candidate_rows = pd.DataFrame(candidates).copy()
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if candidate_rows.empty:
        raise ValueError("risk-aware reservoir CV requires candidate rows")
    if truth_rows.empty:
        raise ValueError("risk-aware reservoir CV requires truth rows")
    if "sequence_id" not in candidate_rows.columns or "sequence_id" not in truth_rows.columns:
        raise ValueError("candidate and truth rows must include sequence_id")
    sequences = sorted(
        set(candidate_rows["sequence_id"].astype(str))
        & set(truth_rows["sequence_id"].astype(str))
    )
    if len(sequences) < 2:
        raise ValueError("at least two sequences are required for risk-aware CV")
    return candidate_rows, truth_rows, sequences


def _validate_risk_parameters(risk_aversion: float, tail_quantile: float) -> None:
    if not np.isfinite(float(risk_aversion)) or not 0.0 <= float(risk_aversion) <= 1.0:
        raise ValueError("risk_aversion must be finite and within [0, 1]")
    if not np.isfinite(float(tail_quantile)) or not 0.0 <= float(tail_quantile) <= 1.0:
        raise ValueError("tail_quantile must be finite and within [0, 1]")


def _quantile_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _json_dict(value: Any) -> dict[str, float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    raw = value if isinstance(value, dict) else json.loads(str(value))
    return {str(key): float(item) for key, item in raw.items()}


def _offset_dict_to_specs(offsets: dict[str, float]) -> list[str]:
    return [f"{key}={value:g}" for key, value in sorted(offsets.items())]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
