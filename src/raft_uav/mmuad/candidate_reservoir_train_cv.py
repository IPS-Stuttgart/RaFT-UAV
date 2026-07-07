"""Train-CV selection for MMUAD candidate-reservoir score offsets.

The reservoir offset grid is useful for diagnosing which raw/dynamic/calibrated
candidate branches should survive into downstream mixture-MAP smoothing.  This
module adds train-only selection wrappers: branch/source score offsets are
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
SELECTION_MODE_PER_FOLD = "per-fold-train-selection"
SELECTION_MODE_CV_AGGREGATE = "cv-aggregate"
SELECTION_MODES = (SELECTION_MODE_PER_FOLD, SELECTION_MODE_CV_AGGREGATE)


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
    """Select reservoir offsets with nested sequence-level cross-validation.

    Each fold selects the best offset config on the remaining training
    sequences, then evaluates only that selected config on the held-out train
    sequence.  The final selected config is fit on all supplied candidates and
    truth.  Fold rows report the leave-one-sequence-out performance of this
    nested selection rule.
    """

    candidate_rows, truth_rows, sequences = _validated_inputs(candidates, truth)

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
    selected_config: dict[str, Any] = _selected_config(
        selection_protocol="leave-one-sequence-out-cv-diagnostic__final-fit-on-all-train",
        sequence_count=len(sequences),
        selection_metric=selection_metric,
        selected_grid_label=str(selected_final.get("grid_label", "")),
        selected_metric_value=float(selected_final.get(selection_metric, float("nan"))),
        selected_branch_offsets=selected_branch_offsets,
        selected_source_offsets=selected_source_offsets,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        global_top_n=global_top_n,
        per_source_top_n=per_source_top_n,
        per_branch_top_n=per_branch_top_n,
        max_candidates_per_frame=max_candidates_per_frame,
        score_floor_quantile=score_floor_quantile,
        top_k_values=top_k_values,
        max_truth_time_delta_s=max_truth_time_delta_s,
    )
    best_reservoir = _selected_reservoir(
        candidate_rows,
        truth_rows,
        selected_branch_offsets=selected_branch_offsets,
        selected_source_offsets=selected_source_offsets,
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
    return selected_config, fold_summary, final_summary, best_reservoir


def select_candidate_reservoir_offsets_by_cv_aggregate(
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
    """Select a single reservoir-offset config by LOSO aggregate performance.

    This mode evaluates every offset-grid row on every held-out training
    sequence and selects the config with the best mean held-out metric.  It is a
    cleaner train-only selector for a fixed validation/test config than the
    per-fold nested diagnostic, which may select different configs in different
    folds.
    """

    candidate_rows, truth_rows, sequences = _validated_inputs(candidates, truth)
    fold_frames: list[pd.DataFrame] = []
    for holdout in sequences:
        holdout_candidates = candidate_rows.loc[candidate_rows["sequence_id"].astype(str) == holdout]
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
        raise ValueError("aggregate CV selection did not produce any fold summaries")
    fold_summary = pd.concat(fold_frames, ignore_index=True)
    aggregate_summary = _aggregate_fold_grid(
        fold_summary,
        selection_metric=selection_metric,
    )
    if aggregate_summary.empty:
        raise ValueError(f"aggregate CV summary missing metric {selection_metric!r}")
    selected = aggregate_summary.iloc[0]
    selected_branch_offsets = _json_dict(selected.get("branch_score_offsets_json", "{}"))
    selected_source_offsets = _json_dict(selected.get("source_score_offsets_json", "{}"))
    best_reservoir = _selected_reservoir(
        candidate_rows,
        truth_rows,
        selected_branch_offsets=selected_branch_offsets,
        selected_source_offsets=selected_source_offsets,
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
    selected_config = _selected_config(
        selection_protocol="leave-one-sequence-out-cv-aggregate__final-apply-selected-offsets",
        sequence_count=len(sequences),
        selection_metric=selection_metric,
        selected_grid_label=str(selected.get("grid_label", "")),
        selected_metric_value=float(selected.get(f"{selection_metric}_mean", float("nan"))),
        selected_branch_offsets=selected_branch_offsets,
        selected_source_offsets=selected_source_offsets,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        global_top_n=global_top_n,
        per_source_top_n=per_source_top_n,
        per_branch_top_n=per_branch_top_n,
        max_candidates_per_frame=max_candidates_per_frame,
        score_floor_quantile=score_floor_quantile,
        top_k_values=top_k_values,
        max_truth_time_delta_s=max_truth_time_delta_s,
    )
    selected_config["selected_metric_std"] = float(selected.get(f"{selection_metric}_std", float("nan")))
    selected_config["selected_metric_fold_count"] = int(selected.get("fold_count", 0))
    return selected_config, fold_summary, aggregate_summary, best_reservoir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-reservoir-train-cv",
        description="select MMUAD candidate-reservoir score offsets on train sequences",
    )
    parser.add_argument("--candidate", action="append", default=[], help="candidate CSV as BRANCH=path")
    parser.add_argument("--candidate-csv", action="append", default=[], help="alias for --candidate")
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default=SELECTION_MODE_PER_FOLD,
        help=(
            "per-fold-train-selection keeps the original nested diagnostic; "
            "cv-aggregate selects one fixed config by mean held-out train metric"
        ),
    )
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
    parser.add_argument("--write-best-reservoir", action="store_true")
    args = parser.parse_args(argv)

    candidate_specs = [*args.candidate, *args.candidate_csv]
    candidates = load_candidate_inputs(candidate_specs)
    truth = pd.read_csv(args.truth_csv)
    selector = (
        select_candidate_reservoir_offsets_by_cv_aggregate
        if args.selection_mode == SELECTION_MODE_CV_AGGREGATE
        else select_candidate_reservoir_offsets_by_sequence_cv
    )
    selected_config, fold_summary, final_summary, best_reservoir = selector(
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
        write_best_reservoir=args.write_best_reservoir,
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
    print(f"selection_mode={args.selection_mode}")
    print(f"selected_config_json={config_json}")
    print(f"fold_summary_csv={fold_csv}")
    print(f"final_grid_summary_csv={final_csv}")
    print(f"selected_grid_label={selected_config['selected_grid_label']}")
    print(f"selected_metric_value={selected_config['selected_metric_value']}")
    return 0


def _validated_inputs(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
    return candidate_rows, truth_rows, sequences


def _selected_config(
    *,
    selection_protocol: str,
    sequence_count: int,
    selection_metric: str,
    selected_grid_label: str,
    selected_metric_value: float,
    selected_branch_offsets: dict[str, float],
    selected_source_offsets: dict[str, float],
    score_column: str,
    fallback_score_column: str,
    global_top_n: int,
    per_source_top_n: int,
    per_branch_top_n: int,
    max_candidates_per_frame: int,
    score_floor_quantile: float | None,
    top_k_values: Sequence[int],
    max_truth_time_delta_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_protocol": selection_protocol,
        "sequence_count": int(sequence_count),
        "selection_metric": selection_metric,
        "selected_grid_label": selected_grid_label,
        "selected_metric_value": float(selected_metric_value),
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


def _selected_reservoir(
    candidate_rows: pd.DataFrame,
    truth_rows: pd.DataFrame,
    *,
    selected_branch_offsets: dict[str, float],
    selected_source_offsets: dict[str, float],
    score_column: str,
    fallback_score_column: str,
    global_top_n: int,
    per_source_top_n: int,
    per_branch_top_n: int,
    max_candidates_per_frame: int,
    score_floor_quantile: float | None,
    top_k_values: Sequence[int],
    max_truth_time_delta_s: float,
    selection_metric: str,
    write_best_reservoir: bool,
) -> pd.DataFrame | None:
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
        write_best_reservoir=write_best_reservoir,
    )
    return best_reservoir if write_best_reservoir else None


def _aggregate_fold_grid(fold_summary: pd.DataFrame, *, selection_metric: str) -> pd.DataFrame:
    if selection_metric not in fold_summary.columns:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for grid_label, group in fold_summary.groupby("grid_label", sort=False):
        values = pd.to_numeric(group[selection_metric], errors="coerce").dropna()
        if values.empty:
            continue
        first = group.iloc[0]
        records.append(
            {
                "grid_label": str(grid_label),
                "branch_score_offsets_json": first.get("branch_score_offsets_json", "{}"),
                "source_score_offsets_json": first.get("source_score_offsets_json", "{}"),
                "fold_count": int(len(values)),
                f"{selection_metric}_mean": float(values.mean()),
                f"{selection_metric}_std": float(values.std(ddof=0)),
                f"{selection_metric}_min": float(values.min()),
                f"{selection_metric}_max": float(values.max()),
            }
        )
    out = pd.DataFrame.from_records(records)
    metric_mean = f"{selection_metric}_mean"
    if not out.empty:
        out = out.sort_values([metric_mean, "grid_label"], na_position="last").reset_index(drop=True)
    return out


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
