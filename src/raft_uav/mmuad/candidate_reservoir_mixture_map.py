"""Run candidate-mixture MAP on a branch-preserving MMUAD reservoir.

The plain candidate-mixture runner ranks candidates inside each frame and can
therefore discard low-scoring raw/dynamic/calibrated branches before the robust
trajectory objective has a chance to use them.  This module wires the
branch-preserving reservoir into mixture-MAP as a single inference-safe command:
first retain a bounded global/source/branch candidate set, then run mixture-MAP
on all retained candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map import LOSS_CHOICES
from raft_uav.mmuad.candidate_mixture_map import run_candidate_mixture_map
from raft_uav.mmuad.candidate_mixture_map import write_candidate_mixture_map_outputs
from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_reservoir import build_candidate_reservoir
from raft_uav.mmuad.candidate_reservoir import build_oracle_recall_tables
from raft_uav.mmuad.candidate_reservoir import build_reservoir_summary
from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import (
    load_sequence_class_map,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)

RESERVOIR_CSV = "mmuad_reservoir_mixture_candidates.csv"
RESERVOIR_SUMMARY_JSON = "mmuad_reservoir_mixture_reservoir_summary.json"
COMBINED_SUMMARY_JSON = "mmuad_reservoir_mixture_summary.json"
RESERVOIR_ORACLE_FRAME_CSV = "mmuad_reservoir_mixture_oracle_frames.csv"
RESERVOIR_ORACLE_SUMMARY_CSV = "mmuad_reservoir_mixture_oracle_summary.csv"
RESERVOIR_ORACLE_BY_SEQUENCE_CSV = "mmuad_reservoir_mixture_oracle_by_sequence.csv"
RESERVOIR_MIXTURE_GAP_SUMMARY_CSV = "mmuad_reservoir_mixture_gap_summary.csv"
RESERVOIR_MIXTURE_GAP_BY_SEQUENCE_CSV = "mmuad_reservoir_mixture_gap_by_sequence.csv"
RESERVOIR_MIXTURE_ASSIGNMENT_BY_BRANCH_CSV = (
    "mmuad_reservoir_mixture_assignment_by_branch.csv"
)
RESERVOIR_MIXTURE_ASSIGNMENT_BY_SOURCE_CSV = (
    "mmuad_reservoir_mixture_assignment_by_source.csv"
)
_DEFAULT_ORACLE_TOP_K = (1, 3, 5, 10, 20)


def run_reservoir_mixture_map(
    candidates: pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    """Build a candidate reservoir and run mixture-MAP over all retained rows."""

    reservoir_config = reservoir_config or ReservoirConfig()
    mixture_config = mixture_config or CandidateMixtureMapConfig()
    reservoir = build_candidate_reservoir(candidates, config=reservoir_config)
    mixture_config = CandidateMixtureMapConfig(
        **{
            **asdict(mixture_config),
            # The reservoir has already bounded the frame candidate set.  Use all
            # retained rows so branch/source-preserved candidates are not pruned
            # again by a single global score ordering.
            "top_k": 0,
        }
    )
    result = run_candidate_mixture_map(
        reservoir,
        config=mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    summary = {
        "schema": "raft-uav-mmuad-reservoir-mixture-map-v1",
        "reservoir_config": asdict(reservoir_config),
        "mixture_config": asdict(mixture_config),
        "reservoir": build_reservoir_summary(candidates, reservoir),
        "mixture": result.summary,
    }
    return reservoir, result, _jsonable(summary)


def build_reservoir_mixture_gap_summary(
    *,
    mixture_summary: dict[str, Any],
    reservoir_oracle_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Compare achieved mixture pose quality with retained-reservoir oracle ceilings.

    The reservoir oracle tells us whether good candidates survived reservoir
    construction.  The achieved mixture metric tells us how well the robust
    trajectory assignment used those retained candidates.  The gap table makes
    that difference explicit for each run.
    """

    pooled_metrics = mixture_summary.get("metrics", {}).get("pooled", {})
    mixture_rmse = _optional_float(pooled_metrics.get("rmse_3d_m"))
    mixture_mse = _optional_float(pooled_metrics.get("mse_3d_m2"))
    if mixture_mse is None and mixture_rmse is not None:
        mixture_mse = mixture_rmse * mixture_rmse
    record: dict[str, Any] = {
        "mixture_mse_3d_m2": mixture_mse,
        "mixture_rmse_3d_m": mixture_rmse,
    }
    if reservoir_oracle_summary.empty:
        return _jsonable(record)

    pooled_oracle = reservoir_oracle_summary.iloc[0].to_dict()
    topk_mse_columns: list[tuple[str, float]] = []
    for column, value in pooled_oracle.items():
        if not str(column).endswith("_mse"):
            continue
        oracle_mse = _optional_float(value)
        if oracle_mse is None:
            continue
        label = _oracle_label_from_mse_column(str(column))
        record[f"reservoir_oracle_{label}_mse_3d_m2"] = oracle_mse
        if mixture_mse is not None:
            record[f"gap_to_oracle_{label}_mse_3d_m2"] = mixture_mse - oracle_mse
            record[f"ratio_to_oracle_{label}_mse"] = _safe_ratio(mixture_mse, oracle_mse)
        if label.startswith("top"):
            topk_mse_columns.append((label, oracle_mse))

    all_mse = _optional_float(record.get("reservoir_oracle_all_mse_3d_m2"))
    if mixture_mse is not None and all_mse is not None:
        record["assignment_gap_mse_3d_m2"] = mixture_mse - all_mse
        record["assignment_gap_ratio"] = _safe_ratio(mixture_mse, all_mse)
    if topk_mse_columns:
        best_label, best_mse = min(topk_mse_columns, key=lambda item: item[1])
        record["best_reservoir_oracle_topk"] = best_label
        record["best_reservoir_oracle_topk_mse_3d_m2"] = best_mse
        if mixture_mse is not None:
            record["gap_to_best_reservoir_oracle_topk_mse_3d_m2"] = mixture_mse - best_mse
            record["ratio_to_best_reservoir_oracle_topk_mse"] = _safe_ratio(
                mixture_mse,
                best_mse,
            )
    return _jsonable(record)


def build_reservoir_mixture_gap_by_sequence(
    *,
    mixture_summary: dict[str, Any],
    reservoir_oracle_by_sequence: pd.DataFrame,
) -> pd.DataFrame:
    """Compare mixture-vs-oracle gaps for each sequence.

    The pooled gap can hide whether one sequence has a damaged reservoir ceiling
    or whether the mixture assignment is weak across many sequences.  This table
    makes the same assignment/oracle gap diagnostic available at sequence level.
    """

    if reservoir_oracle_by_sequence.empty:
        return pd.DataFrame()
    sequence_metrics = mixture_summary.get("metrics", {}).get("sequences", {})
    records: list[dict[str, Any]] = []
    for _, oracle_row in reservoir_oracle_by_sequence.iterrows():
        sequence_id = str(oracle_row.get("sequence_id"))
        metrics = sequence_metrics.get(sequence_id, {})
        mixture_rmse = _optional_float(metrics.get("rmse_3d_m"))
        mixture_mse = _optional_float(metrics.get("mse_3d_m2"))
        if mixture_mse is None and mixture_rmse is not None:
            mixture_mse = mixture_rmse * mixture_rmse
        record: dict[str, Any] = {
            "sequence_id": sequence_id,
            "mixture_mse_3d_m2": mixture_mse,
            "mixture_rmse_3d_m": mixture_rmse,
        }
        topk_mse_columns: list[tuple[str, float]] = []
        for column, value in oracle_row.to_dict().items():
            if not str(column).endswith("_mse"):
                continue
            oracle_mse = _optional_float(value)
            if oracle_mse is None:
                continue
            label = _oracle_label_from_mse_column(str(column))
            record[f"reservoir_oracle_{label}_mse_3d_m2"] = oracle_mse
            if mixture_mse is not None:
                record[f"gap_to_oracle_{label}_mse_3d_m2"] = mixture_mse - oracle_mse
                record[f"ratio_to_oracle_{label}_mse"] = _safe_ratio(mixture_mse, oracle_mse)
            if label.startswith("top"):
                topk_mse_columns.append((label, oracle_mse))
        all_mse = _optional_float(record.get("reservoir_oracle_all_mse_3d_m2"))
        if mixture_mse is not None and all_mse is not None:
            record["assignment_gap_mse_3d_m2"] = mixture_mse - all_mse
            record["assignment_gap_ratio"] = _safe_ratio(mixture_mse, all_mse)
        if topk_mse_columns:
            best_label, best_mse = min(topk_mse_columns, key=lambda item: item[1])
            record["best_reservoir_oracle_topk"] = best_label
            record["best_reservoir_oracle_topk_mse_3d_m2"] = best_mse
            if mixture_mse is not None:
                record["gap_to_best_reservoir_oracle_topk_mse_3d_m2"] = mixture_mse - best_mse
                record["ratio_to_best_reservoir_oracle_topk_mse"] = _safe_ratio(
                    mixture_mse,
                    best_mse,
                )
        records.append(_jsonable(record))
    return pd.DataFrame.from_records(records)


def build_assignment_usage_summary(assignments: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Summarize mixture responsibility and dominant assignments by branch/source.

    This is inference-safe: it does not use truth. It helps diagnose whether a
    reservoir preserved multiple candidate branches only nominally, or whether
    the mixture objective actually assigned probability mass to them.
    """

    rows = pd.DataFrame(assignments).copy()
    if rows.empty:
        return pd.DataFrame()
    if group_column not in rows.columns:
        rows[group_column] = "unknown"
    rows[group_column] = rows[group_column].fillna("unknown").astype(str)
    total_frames = _frame_count(rows)
    total_responsibility = _numeric_series(rows, "mixture_final_weight", 0.0).sum()
    records: list[dict[str, Any]] = []
    for value, group in rows.groupby(group_column, sort=True):
        responsibility = _numeric_series(group, "mixture_final_weight", 0.0)
        candidate_rank = _numeric_series(group, "candidate_rank", float("nan"))
        sigma = _numeric_series(group, "mixture_sigma_m", float("nan"))
        distance = _numeric_series(group, "mixture_distance_to_state_m", float("nan"))
        dominant = (
            group["mixture_dominant"].fillna(False).astype(bool)
            if "mixture_dominant" in group.columns
            else pd.Series(False, index=group.index)
        )
        responsibility_sum = float(responsibility.sum())
        record = {
            group_column: str(value),
            "candidate_rows": int(len(group)),
            "frame_count": int(_frame_count(group)),
            "frame_coverage_fraction": _safe_ratio(_frame_count(group), total_frames),
            "responsibility_sum": responsibility_sum,
            "responsibility_fraction_of_total": _safe_ratio(
                responsibility_sum,
                float(total_responsibility),
            ),
            "responsibility_mean": _series_mean(responsibility),
            "dominant_count": int(dominant.sum()),
            "dominant_fraction_of_frames": _safe_ratio(int(dominant.sum()), total_frames),
            "dominant_fraction_of_group_rows": _safe_ratio(int(dominant.sum()), len(group)),
            "mean_candidate_rank": _series_mean(candidate_rank),
            "p95_candidate_rank": _series_quantile(candidate_rank, 0.95),
            "mean_sigma_m": _series_mean(sigma),
            "mean_distance_to_state_m": _series_mean(distance),
        }
        records.append(_jsonable(record))
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame.from_records(records)
    return out.sort_values(
        ["responsibility_fraction_of_total", "dominant_count", group_column],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def write_reservoir_mixture_map_outputs(
    *,
    reservoir: pd.DataFrame,
    result: Any,
    summary: dict[str, Any],
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 2,
    official_results_csv: Path | None = None,
    official_zip: Path | None = None,
) -> dict[str, Path]:
    """Write reservoir rows, mixture artifacts, summaries, and optional upload files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = write_candidate_mixture_map_outputs(result, output)
    reservoir_csv = output / RESERVOIR_CSV
    reservoir.to_csv(reservoir_csv, index=False)
    paths["reservoir_csv"] = reservoir_csv
    reservoir_summary_json = output / RESERVOIR_SUMMARY_JSON
    reservoir_summary_json.write_text(
        json.dumps(_jsonable(summary.get("reservoir", {})), indent=2),
        encoding="utf-8",
    )
    paths["reservoir_summary_json"] = reservoir_summary_json
    branch_usage = build_assignment_usage_summary(result.assignments, "candidate_branch")
    source_usage = build_assignment_usage_summary(result.assignments, "source")
    branch_usage_csv = output / RESERVOIR_MIXTURE_ASSIGNMENT_BY_BRANCH_CSV
    source_usage_csv = output / RESERVOIR_MIXTURE_ASSIGNMENT_BY_SOURCE_CSV
    branch_usage.to_csv(branch_usage_csv, index=False)
    source_usage.to_csv(source_usage_csv, index=False)
    paths["assignment_by_branch_csv"] = branch_usage_csv
    paths["assignment_by_source_csv"] = source_usage_csv
    summary["assignment_usage"] = {
        "branch_count": int(len(branch_usage)),
        "source_count": int(len(source_usage)),
        "top_branch_by_responsibility": _max_record(
            branch_usage,
            "responsibility_fraction_of_total",
        ),
        "top_source_by_responsibility": _max_record(
            source_usage,
            "responsibility_fraction_of_total",
        ),
    }
    combined_summary_json = output / COMBINED_SUMMARY_JSON
    combined_summary_json.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    paths["combined_summary_json"] = combined_summary_json
    class_map = class_map or {}
    if official_results_csv is not None:
        write_official_mmaud_results_csv(
            result.estimates,
            official_results_csv,
            classification=default_classification,
            class_map=class_map,
        )
        paths["official_results_csv"] = Path(official_results_csv)
    if official_zip is not None:
        write_official_ug2_codabench_zip(
            result.estimates,
            official_zip,
            classification=default_classification,
            class_map=class_map,
        )
        paths["official_zip"] = Path(official_zip)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-reservoir-mixture-map",
        description="run MMUAD candidate-mixture MAP over a branch-preserving reservoir",
    )
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="BRANCH=PATH",
        help="candidate CSV to include; may be repeated; plain PATH uses file stem as branch",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--reservoir-score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--reservoir-fallback-score-column", default="confidence")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument(
        "--reservoir-cap-reason-bonus",
        type=float,
        default=0.0,
        help=(
            "bonus added during the final reservoir cap for candidates selected by "
            "multiple independent rules"
        ),
    )
    parser.add_argument("--mixture-score-column", default="candidate_reservoir_score")
    parser.add_argument("--mixture-fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="2")
    parser.add_argument("--official-results-csv", type=Path)
    parser.add_argument("--official-zip", type=Path)
    parser.add_argument(
        "--reservoir-oracle-frame-csv",
        type=Path,
        help="optional reservoir oracle recall rows; defaults inside --output-dir when truth is supplied",
    )
    parser.add_argument(
        "--reservoir-oracle-summary-csv",
        type=Path,
        help="optional pooled reservoir oracle recall summary",
    )
    parser.add_argument(
        "--reservoir-oracle-by-sequence-csv",
        type=Path,
        help="optional per-sequence reservoir oracle recall summary",
    )
    parser.add_argument(
        "--reservoir-mixture-gap-summary-csv",
        type=Path,
        help="optional mixture-vs-reservoir-oracle gap summary",
    )
    parser.add_argument(
        "--reservoir-mixture-gap-by-sequence-csv",
        type=Path,
        help="optional per-sequence mixture-vs-reservoir-oracle gap summary",
    )
    parser.add_argument("--oracle-top-k", type=int, action="append", default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    candidates = load_candidate_inputs(args.candidate_csv)
    truth = None if args.truth_csv is None else load_evaluation_truth_file(args.truth_csv).rows
    initial_estimates = None
    if args.initial_estimates_csv is not None:
        initial_estimates = pd.read_csv(args.initial_estimates_csv)
    reservoir_config = ReservoirConfig(
        global_top_n=int(args.global_top_n),
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        max_candidates_per_frame=int(args.max_candidates_per_frame),
        score_column=str(args.reservoir_score_column),
        fallback_score_column=str(args.reservoir_fallback_score_column),
        score_floor_quantile=args.score_floor_quantile,
        cap_reason_bonus=float(args.reservoir_cap_reason_bonus),
    )
    mixture_config = CandidateMixtureMapConfig(
        top_k=0,
        score_column=str(args.mixture_score_column),
        fallback_score_columns=tuple(args.mixture_fallback_score_column)
        or ("ranker_score", "confidence"),
        sigma_column=str(args.sigma_column),
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        score_weight=float(args.score_weight),
        temperature=float(args.temperature),
        sigma_log_weight=float(args.sigma_log_weight),
        loss=str(args.loss),
        huber_delta=float(args.huber_delta),
        smoothness_weight=float(args.smoothness_weight),
        iterations=int(args.iterations),
        uniform_weight_floor=float(args.uniform_weight_floor),
        branch_balance=float(args.branch_balance),
        source_balance=float(args.source_balance),
        responsibility_floor=float(args.responsibility_floor),
    )
    reservoir, result, summary = run_reservoir_mixture_map(
        candidates,
        reservoir_config=reservoir_config,
        mixture_config=mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    diagnostic_paths: dict[str, Path] = {}
    if truth is not None:
        top_k_values = (
            tuple(args.oracle_top_k) if args.oracle_top_k is not None else _DEFAULT_ORACLE_TOP_K
        )
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            reservoir,
            truth,
            top_k_values=top_k_values,
            max_truth_time_delta_s=float(args.max_truth_time_delta_s),
        )
        summary["reservoir_oracle"] = {
            "top_k_values": list(top_k_values),
            "max_truth_time_delta_s": float(args.max_truth_time_delta_s),
            "frame_count": int(len(frame_rows)),
            "pooled": _first_record(pooled),
        }
        gap_summary = build_reservoir_mixture_gap_summary(
            mixture_summary=result.summary,
            reservoir_oracle_summary=pooled,
        )
        gap_by_sequence = build_reservoir_mixture_gap_by_sequence(
            mixture_summary=result.summary,
            reservoir_oracle_by_sequence=by_sequence,
        )
        summary["reservoir_mixture_gap"] = gap_summary
        summary["reservoir_mixture_gap_by_sequence"] = {
            "sequence_count": int(len(gap_by_sequence)),
            "worst_assignment_gap": _max_record(
                gap_by_sequence,
                "assignment_gap_mse_3d_m2",
            ),
        }
        frame_path = args.reservoir_oracle_frame_csv or args.output_dir / RESERVOIR_ORACLE_FRAME_CSV
        pooled_path = args.reservoir_oracle_summary_csv or args.output_dir / RESERVOIR_ORACLE_SUMMARY_CSV
        by_sequence_path = (
            args.reservoir_oracle_by_sequence_csv
            or args.output_dir / RESERVOIR_ORACLE_BY_SEQUENCE_CSV
        )
        gap_path = (
            args.reservoir_mixture_gap_summary_csv
            or args.output_dir / RESERVOIR_MIXTURE_GAP_SUMMARY_CSV
        )
        gap_by_sequence_path = (
            args.reservoir_mixture_gap_by_sequence_csv
            or args.output_dir / RESERVOIR_MIXTURE_GAP_BY_SEQUENCE_CSV
        )
        _write_frame(frame_rows, frame_path)
        _write_frame(pooled, pooled_path)
        _write_frame(by_sequence, by_sequence_path)
        _write_frame(pd.DataFrame.from_records([gap_summary]), gap_path)
        _write_frame(gap_by_sequence, gap_by_sequence_path)
        diagnostic_paths = {
            "reservoir_oracle_frame_csv": frame_path,
            "reservoir_oracle_summary_csv": pooled_path,
            "reservoir_oracle_by_sequence_csv": by_sequence_path,
            "reservoir_mixture_gap_summary_csv": gap_path,
            "reservoir_mixture_gap_by_sequence_csv": gap_by_sequence_path,
        }
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_reservoir_mixture_map_outputs(
        reservoir=reservoir,
        result=result,
        summary=summary,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        official_results_csv=args.official_results_csv,
        official_zip=args.official_zip,
    )
    paths.update(diagnostic_paths)
    print("mmuad_reservoir_mixture_map=ok")
    print(f"reservoir_rows={len(reservoir)}")
    print(f"estimate_rows={len(result.estimates)}")
    pooled = result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _first_record(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return _jsonable(frame.iloc[0].to_dict())


def _max_record(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {}
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return {}
    return _jsonable(frame.loc[int(values.idxmax())].to_dict())


def _frame_count(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    if {"sequence_id", "time_s"}.issubset(rows.columns):
        return int(len(rows[["sequence_id", "time_s"]].drop_duplicates()))
    return 0


def _numeric_series(rows: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(default, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _series_mean(values: pd.Series) -> float | None:
    finite = values.dropna()
    if finite.empty:
        return None
    return float(finite.mean())


def _series_quantile(values: pd.Series, quantile: float) -> float | None:
    finite = values.dropna()
    if finite.empty:
        return None
    return float(finite.quantile(quantile))


def _oracle_label_from_mse_column(column: str) -> str:
    if column == "oracle_all_3d_m_mse":
        return "all"
    if column.startswith("oracle_") and column.endswith("_3d_m_mse"):
        return column.removeprefix("oracle_").removesuffix("_3d_m_mse")
    return column.removesuffix("_mse")


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


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
