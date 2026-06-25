"""Train-CV selection for MMUAD candidate-reservoir score offsets.

The reservoir offset grid is useful for diagnosing which raw/dynamic/calibrated
candidate branches should survive into downstream mixture-MAP smoothing.  This
module adds a train-only selection wrapper: branch/source score offsets are
selected on training sequences, evaluated with leave-one-sequence-out folds, and
written as a reusable config JSON that can be applied to validation/test without
truth labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_grid import run_candidate_reservoir_offset_grid
from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs
from raft_uav.mmuad.schema import normalize_truth_columns

_DEFAULT_TOP_K = (1, 3, 5, 10, 20)
_DEFAULT_SELECTION_METRIC = "oracle_top5_3d_m_mse"


def select_candidate_reservoir_offsets_by_sequence_cv(
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
    write_best_reservoir: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Select reservoir offsets with sequence-level cross-validation.

    The final selected config is fit on all supplied candidates/truth.  Fold rows
    report leave-one-sequence-out performance of the same selection rule, so the
    JSON can be used as a train-selected config for downstream validation/test
    runs.
    """

    candidate_rows = pd.DataFrame(candidates).copy()
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if candidate_rows.empty:
        raise ValueError("candidate reservoir train-CV selection requires candidate rows")
    if truth_rows.empty:
        raise ValueError("candidate reservoir train-CV selection requires truth rows")
    if "sequence_id" not in candidate_rows.columns or "sequence_id" not in truth_rows.columns:
        raise ValueError("candidate and truth rows must include sequence_id")

    sequences = sorted(
        set(candidate_rows["sequence_id"].astype(str)) & set(truth_rows["sequence_id"].astype(str))
    )
    if len(sequences) < 2:
        raise ValueError("at least two sequences are required for leave-one-sequence-out CV")

    fold_records: list[dict[str, Any]] = []
    for holdout in sequences:
        train_candidates = candidate_rows.loc[candidate_rows["sequence_id"].astype(str) != holdout]
        holdout_candidates = candidate_rows.loc[candidate_rows["sequence_id"].astype(str) == holdout]
        train_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) != holdout]
        holdout_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) == holdout]
        if train_candidates.empty or holdout_candidates.empty or train_truth.empty or holdout_truth.empty:
            continue
        train_summary, _ = run_candidate_reservoir_offset_grid(
            train_candidates,
            truth=train_truth,
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
        if train_summary.empty:
            continue
        selected = train_summary.iloc[0]
        branch_offsets = _json_dict(selected.get("branch_score_offsets_json", "{}"))
        source_offsets = _json_dict(selected.get("source_score_offsets_json", "{}"))
        holdout_summary, _ = run_candidate_reservoir_offset_grid(
            holdout_candidates,
            truth=holdout_truth,
            branch_offset_grid=_offset_dict_to_specs(branch_offsets),
            source_offset_grid=_offset_dict_to_specs(source_offsets),
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
        holdout_record = holdout_summary.iloc[0].to_dict()
        holdout_record.update(
            {
                "holdout_sequence_id": holdout,
                "selected_grid_label": str(selected.get("grid_label", "")),
                "selected_branch_score_offsets_json": json.dumps(branch_offsets, sort_keys=True),
                "selected_source_score_offsets_json": json.dumps(source_offsets, sort_keys=True),
                "train_selection_metric": selection_metric,
                "train_selection_metric_value": float(selected.get(selection_metric, float("nan"))),
            }
        )
        fold_records.append(holdout_record)

    fold_summary = pd.DataFrame.from_records(fold_records)
    final_summary, _ = run_candidate_reservoir_offset_grid(
        candidate_rows,
        truth=truth_rows,
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
    if final_summary.empty:
        raise ValueError("final train grid did not produce any reservoir summaries")
    selected_final = final_summary.iloc[0]
    selected_branch_offsets = _json_dict(selected_final.get("branch_score_offsets_json", "{}"))
    selected_source_offsets = _json_dict(selected_final.get("source_score_offsets_json", "{}"))
    selected_config: dict[str, Any] = {
        "schema_version": 1,
        "selection_protocol": "leave-one-sequence-out-cv-diagnostic__final-fit-on-all-train",
        "sequence_count": len(sequences),
        "selection_metric": selection_metric,
        "selected_grid_label": str(selected_final.get("grid_label", "")),
        "selected_metric_value": float(selected_final.get(selection_metric, float("nan"))),
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
    }
    best_reservoir = None
    if write_best_reservoir:
        _, best_reservoir = run_candidate_reservoir_offset_grid(
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
            write_best_reservoir=True,
        )
    return selected_config, fold_summary, final_summary, best_reservoir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-reservoir-train-cv",
        description="select MMUAD candidate-reservoir score offsets on train sequences",
    )
    parser.add_argument("--candidate", action="append", default=[], help="candidate CSV as BRANCH=path")
    parser.add_argument("--candidate-csv", action="append", default=[], help="alias for --candidate")
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
    parser.add_argument("--top-k", action="append", type=int, default=list(_DEFAULT_TOP_K))
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--selection-metric", default=_DEFAULT_SELECTION_METRIC)
    parser.add_argument("--write-best-reservoir", action="store_true")
    args = parser.parse_args(argv)

    candidate_specs = [*args.candidate, *args.candidate_csv]
    candidates = load_candidate_inputs(candidate_specs)
    truth = pd.read_csv(args.truth_csv)
    selected_config, fold_summary, final_summary, best_reservoir = (
        select_candidate_reservoir_offsets_by_sequence_cv(
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
            top_k_values=tuple(args.top_k),
            max_truth_time_delta_s=args.max_truth_time_delta_s,
            selection_metric=args.selection_metric,
            write_best_reservoir=args.write_best_reservoir,
        )
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_json = output_dir / "mmuad_candidate_reservoir_train_selected_config.json"
    fold_csv = output_dir / "mmuad_candidate_reservoir_train_cv_folds.csv"
    final_csv = output_dir / "mmuad_candidate_reservoir_train_final_grid_summary.csv"
    config_json.write_text(json.dumps(selected_config, indent=2), encoding="utf-8")
    fold_summary.to_csv(fold_csv, index=False)
    final_summary.to_csv(final_csv, index=False)
    if best_reservoir is not None:
        best_reservoir.to_csv(output_dir / "mmuad_candidate_reservoir_train_selected.csv", index=False)
    print("mmuad_candidate_reservoir_train_cv=ok")
    print(f"selected_config_json={config_json}")
    print(f"fold_summary_csv={fold_csv}")
    print(f"final_grid_summary_csv={final_csv}")
    print(f"selected_grid_label={selected_config['selected_grid_label']}")
    print(f"selected_metric_value={selected_config['selected_metric_value']}")
    return 0


def _json_dict(value: Any) -> dict[str, float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        raw = value
    else:
        raw = json.loads(str(value))
    return {str(key): float(val) for key, val in raw.items()}


def _offset_dict_to_specs(offsets: dict[str, float]) -> list[str]:
    return [f"{key}={value:g}" for key, value in sorted(offsets.items())]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
