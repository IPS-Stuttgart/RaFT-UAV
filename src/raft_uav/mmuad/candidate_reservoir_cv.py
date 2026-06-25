"""Select MMUAD candidate-reservoir score offsets using train-only LOSO.

The public-validation reservoir offset grid is useful diagnostically, but a
paper or hidden-test pipeline must freeze those offsets without looking at
validation truth. This module evaluates each branch/source offset configuration
per training sequence, performs leave-one-sequence-out selection, and writes a
frozen full-train configuration for later validation/test application.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs
from raft_uav.mmuad.candidate_reservoir_grid import run_candidate_reservoir_offset_grid
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns

SCHEMA = "raft-uav-mmuad-candidate-reservoir-offset-loso-v1"
DEFAULT_SELECTION_METRIC = "oracle_top5_3d_m_mse"
DEFAULT_TOP_K = (1, 3, 5, 10, 20)


def select_candidate_reservoir_offsets_loso(
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
    top_k_values: Sequence[int] = DEFAULT_TOP_K,
    max_truth_time_delta_s: float = 0.5,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    build_selected_train_reservoir: bool = False,
) -> dict[str, Any]:
    """Select branch/source score offsets using leave-one-sequence-out CV."""

    candidate_rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if candidate_rows.empty:
        raise ValueError("LOSO reservoir selection requires candidate rows")
    if truth_rows.empty:
        raise ValueError("LOSO reservoir selection requires truth rows")

    candidate_rows["_sequence_key"] = candidate_rows["sequence_id"].astype(str)
    truth_rows["_sequence_key"] = truth_rows["sequence_id"].astype(str)
    sequence_ids = sorted(
        set(candidate_rows["_sequence_key"]).intersection(truth_rows["_sequence_key"])
    )
    if len(sequence_ids) < 2:
        raise ValueError("LOSO reservoir selection requires at least two shared sequences")

    top_k_tuple = tuple(sorted({int(value) for value in top_k_values if int(value) > 0}))
    per_sequence_parts: list[pd.DataFrame] = []
    for sequence_id in sequence_ids:
        sequence_candidates = candidate_rows.loc[
            candidate_rows["_sequence_key"] == sequence_id
        ].drop(columns=["_sequence_key"])
        sequence_truth = truth_rows.loc[truth_rows["_sequence_key"] == sequence_id].drop(
            columns=["_sequence_key"]
        )
        summary, _ = run_candidate_reservoir_offset_grid(
            sequence_candidates,
            truth=sequence_truth,
            branch_offset_grid=branch_offset_grid,
            source_offset_grid=source_offset_grid,
            score_column=score_column,
            fallback_score_column=fallback_score_column,
            global_top_n=global_top_n,
            per_source_top_n=per_source_top_n,
            per_branch_top_n=per_branch_top_n,
            max_candidates_per_frame=max_candidates_per_frame,
            score_floor_quantile=score_floor_quantile,
            top_k_values=top_k_tuple,
            max_truth_time_delta_s=max_truth_time_delta_s,
            selection_metric=selection_metric,
        )
        if selection_metric not in summary.columns:
            raise ValueError(
                f"selection metric {selection_metric!r} was not produced for {sequence_id}"
            )
        summary = summary.copy()
        summary.insert(0, "sequence_id", sequence_id)
        summary["selection_metric_value"] = pd.to_numeric(
            summary[selection_metric], errors="coerce"
        )
        if not np.isfinite(summary["selection_metric_value"]).any():
            raise ValueError(
                f"selection metric {selection_metric!r} has no finite values for {sequence_id}"
            )
        per_sequence_parts.append(summary)

    per_sequence = pd.concat(per_sequence_parts, ignore_index=True)
    config_summary = _aggregate_config_summary(per_sequence)
    fold_summary = _build_loso_fold_summary(
        per_sequence,
        sequence_ids=sequence_ids,
        selection_metric=selection_metric,
    )
    selection_counts = fold_summary["selected_grid_label"].value_counts()
    config_summary["loso_selected_fold_count"] = (
        config_summary["grid_label"].map(selection_counts).fillna(0).astype(int)
    )

    full_candidates = candidate_rows.drop(columns=["_sequence_key"])
    full_truth = truth_rows.drop(columns=["_sequence_key"])
    full_grid, _ = run_candidate_reservoir_offset_grid(
        full_candidates,
        truth=full_truth,
        branch_offset_grid=branch_offset_grid,
        source_offset_grid=source_offset_grid,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        global_top_n=global_top_n,
        per_source_top_n=per_source_top_n,
        per_branch_top_n=per_branch_top_n,
        max_candidates_per_frame=max_candidates_per_frame,
        score_floor_quantile=score_floor_quantile,
        top_k_values=top_k_tuple,
        max_truth_time_delta_s=max_truth_time_delta_s,
        selection_metric=selection_metric,
    )
    if selection_metric in full_grid.columns:
        pooled_map = pd.Series(
            pd.to_numeric(full_grid[selection_metric], errors="coerce").to_numpy(),
            index=full_grid["grid_label"].astype(str),
        )
        config_summary["full_train_pooled_metric"] = config_summary["grid_label"].map(
            pooled_map
        )

    config_summary = config_summary.sort_values(
        ["train_sequence_mean_metric", "train_sequence_max_metric", "grid_label"],
        na_position="last",
    ).reset_index(drop=True)
    selected_row = config_summary.iloc[0]
    branch_offsets = json.loads(str(selected_row["branch_score_offsets_json"]))
    source_offsets = json.loads(str(selected_row["source_score_offsets_json"]))

    selected_train_reservoir: pd.DataFrame | None = None
    if build_selected_train_reservoir:
        _, selected_train_reservoir = run_candidate_reservoir_offset_grid(
            full_candidates,
            branch_offset_grid=_single_offset_specs(branch_offsets),
            source_offset_grid=_single_offset_specs(source_offsets),
            score_column=score_column,
            fallback_score_column=fallback_score_column,
            global_top_n=global_top_n,
            per_source_top_n=per_source_top_n,
            per_branch_top_n=per_branch_top_n,
            max_candidates_per_frame=max_candidates_per_frame,
            score_floor_quantile=score_floor_quantile,
            top_k_values=top_k_tuple,
            max_truth_time_delta_s=max_truth_time_delta_s,
            selection_metric=selection_metric,
            write_best_reservoir=True,
        )

    heldout_values = pd.to_numeric(fold_summary["heldout_metric_value"], errors="coerce")
    selected_config = {
        "schema": SCHEMA,
        "protocol": "train_sequence_loso_oracle_recall_then_full_train_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": selection_metric,
        "sequence_count": len(sequence_ids),
        "sequence_ids": sequence_ids,
        "selected_grid_label": str(selected_row["grid_label"]),
        "branch_score_offsets": branch_offsets,
        "source_score_offsets": source_offsets,
        "reservoir": {
            "score_column": score_column,
            "fallback_score_column": fallback_score_column,
            "global_top_n": int(global_top_n),
            "per_source_top_n": int(per_source_top_n),
            "per_branch_top_n": int(per_branch_top_n),
            "max_candidates_per_frame": int(max_candidates_per_frame),
            "score_floor_quantile": score_floor_quantile,
            "top_k_values": list(top_k_tuple),
            "max_truth_time_delta_s": float(max_truth_time_delta_s),
        },
        "train_sequence_mean_metric": _finite_or_none(
            selected_row["train_sequence_mean_metric"]
        ),
        "train_sequence_median_metric": _finite_or_none(
            selected_row["train_sequence_median_metric"]
        ),
        "train_sequence_max_metric": _finite_or_none(
            selected_row["train_sequence_max_metric"]
        ),
        "full_train_pooled_metric": _finite_or_none(
            selected_row.get("full_train_pooled_metric")
        ),
        "loso_mean_heldout_metric": _finite_or_none(heldout_values.mean()),
        "loso_median_heldout_metric": _finite_or_none(heldout_values.median()),
        "loso_max_heldout_metric": _finite_or_none(heldout_values.max()),
    }
    return {
        "selected_config": selected_config,
        "per_sequence_summary": per_sequence,
        "fold_summary": fold_summary,
        "config_summary": config_summary,
        "selected_train_reservoir": selected_train_reservoir,
    }


def write_candidate_reservoir_offset_loso_outputs(
    result: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Write LOSO selection artifacts and return their paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected_config_json": output / "mmuad_candidate_reservoir_offset_selected_config.json",
        "per_sequence_csv": output / "mmuad_candidate_reservoir_offset_loso_by_sequence.csv",
        "fold_summary_csv": output / "mmuad_candidate_reservoir_offset_loso_fold_summary.csv",
        "config_summary_csv": output / "mmuad_candidate_reservoir_offset_loso_config_summary.csv",
    }
    paths["selected_config_json"].write_text(
        json.dumps(_jsonable(result["selected_config"]), indent=2),
        encoding="utf-8",
    )
    result["per_sequence_summary"].to_csv(paths["per_sequence_csv"], index=False)
    result["fold_summary"].to_csv(paths["fold_summary_csv"], index=False)
    result["config_summary"].to_csv(paths["config_summary_csv"], index=False)
    reservoir = result.get("selected_train_reservoir")
    if isinstance(reservoir, pd.DataFrame):
        reservoir_path = output / "selected_train_candidate_reservoir.csv"
        reservoir.to_csv(reservoir_path, index=False)
        paths["selected_train_reservoir_csv"] = reservoir_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-select-reservoir-offsets",
        description="select MMUAD reservoir branch/source offsets with train-only LOSO",
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
    parser.add_argument("--top-k", action="append", type=int, default=list(DEFAULT_TOP_K))
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--selection-metric", default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--write-selected-train-reservoir", action="store_true")
    args = parser.parse_args(argv)

    candidate_specs = [*args.candidate, *args.candidate_csv]
    candidates = load_candidate_inputs(candidate_specs)
    if candidates.empty:
        raise ValueError("at least one non-empty --candidate BRANCH=PATH CSV is required")
    result = select_candidate_reservoir_offsets_loso(
        candidates,
        pd.read_csv(args.truth_csv),
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
        build_selected_train_reservoir=args.write_selected_train_reservoir,
    )
    paths = write_candidate_reservoir_offset_loso_outputs(result, output_dir=args.output_dir)
    selected = result["selected_config"]
    print("mmuad_candidate_reservoir_offset_loso=ok")
    print(f"selected_grid_label={selected['selected_grid_label']}")
    print(f"selection_metric={selected['selection_metric']}")
    print(f"loso_mean_heldout_metric={selected['loso_mean_heldout_metric']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _aggregate_config_summary(per_sequence: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "grid_label",
        "branch_score_offsets_json",
        "source_score_offsets_json",
    ]
    grouped = per_sequence.groupby(grouping, dropna=False)["selection_metric_value"]
    summary = grouped.agg(["mean", "median", "max", "count"]).reset_index()
    return summary.rename(
        columns={
            "mean": "train_sequence_mean_metric",
            "median": "train_sequence_median_metric",
            "max": "train_sequence_max_metric",
            "count": "train_sequence_valid_count",
        }
    )


def _build_loso_fold_summary(
    per_sequence: pd.DataFrame,
    *,
    sequence_ids: Sequence[str],
    selection_metric: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for heldout_sequence in sequence_ids:
        training = per_sequence.loc[per_sequence["sequence_id"] != heldout_sequence]
        ranking = (
            training.groupby("grid_label", dropna=False)["selection_metric_value"]
            .agg(["mean", "max", "count"])
            .reset_index()
        )
        ranking = ranking.loc[np.isfinite(pd.to_numeric(ranking["mean"], errors="coerce"))]
        if ranking.empty:
            raise ValueError(f"no finite LOSO training metrics for held-out {heldout_sequence}")
        ranking = ranking.sort_values(["mean", "max", "grid_label"]).reset_index(drop=True)
        selected_label = str(ranking.iloc[0]["grid_label"])
        heldout_rows = per_sequence.loc[
            (per_sequence["sequence_id"] == heldout_sequence)
            & (per_sequence["grid_label"].astype(str) == selected_label)
        ]
        if heldout_rows.empty:
            raise ValueError(
                f"selected grid label {selected_label!r} missing for held-out {heldout_sequence}"
            )
        heldout = heldout_rows.iloc[0]
        records.append(
            {
                "heldout_sequence_id": heldout_sequence,
                "selection_metric": selection_metric,
                "selected_grid_label": selected_label,
                "branch_score_offsets_json": heldout["branch_score_offsets_json"],
                "source_score_offsets_json": heldout["source_score_offsets_json"],
                "train_sequence_mean_metric": float(ranking.iloc[0]["mean"]),
                "train_sequence_max_metric": float(ranking.iloc[0]["max"]),
                "train_sequence_count": int(ranking.iloc[0]["count"]),
                "heldout_metric_value": _finite_or_none(
                    heldout["selection_metric_value"]
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _single_offset_specs(offsets: dict[str, float]) -> list[str]:
    return [f"{name}={float(value):g}" for name, value in sorted(offsets.items())]


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


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
