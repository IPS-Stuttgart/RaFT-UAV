"""Sweep branch/source score offsets for MMUAD candidate reservoirs.

The branch-preserving reservoir keeps raw, dynamic, translated, and merged
candidate streams alive. This companion CLI searches simple additive branch or
source score priors before reservoir selection, then writes oracle-recall
summaries when truth/reference rows are available.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_candidate_reservoir,
    build_oracle_recall_tables,
    build_reservoir_summary,
    load_candidate_inputs,
)
from raft_uav.mmuad.schema import normalize_truth_columns

_DEFAULT_TOP_K = (1, 3, 5, 10, 20)
_DEFAULT_SELECTION_METRIC = "oracle_top5_3d_m_mse"


def run_candidate_reservoir_offset_grid(
    candidates: pd.DataFrame,
    *,
    truth: pd.DataFrame | None = None,
    branch_offset_grid: Sequence[str] = (),
    source_offset_grid: Sequence[str] = (),
    output_dir: Path | None = None,
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
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run a branch/source score-offset reservoir grid."""

    rows = pd.DataFrame(candidates).copy()
    if rows.empty:
        raise ValueError("candidate reservoir offset grid requires candidate rows")
    truth_rows = None if truth is None else normalize_truth_columns(pd.DataFrame(truth).copy())
    offset_configs = _offset_config_grid(branch_offset_grid, source_offset_grid)
    top_k_tuple = tuple(sorted({int(value) for value in top_k_values if int(value) > 0}))
    summary_records: list[dict[str, Any]] = []
    reservoirs: dict[str, pd.DataFrame] = {}
    frame_tables: dict[str, pd.DataFrame] = {}
    by_sequence_tables: dict[str, pd.DataFrame] = {}
    for index, (label, branch_offsets, source_offsets) in enumerate(offset_configs, start=1):
        adjusted = _with_adjusted_scores(
            rows,
            branch_offsets=branch_offsets,
            source_offsets=source_offsets,
            score_column=score_column,
            fallback_score_column=fallback_score_column,
        )
        reservoir = build_candidate_reservoir(
            adjusted,
            config=ReservoirConfig(
                global_top_n=int(global_top_n),
                per_source_top_n=int(per_source_top_n),
                per_branch_top_n=int(per_branch_top_n),
                max_candidates_per_frame=int(max_candidates_per_frame),
                score_column="candidate_reservoir_grid_score",
                fallback_score_column=fallback_score_column,
                score_floor_quantile=score_floor_quantile,
            ),
        )
        summary: dict[str, Any] = {
            "grid_index": int(index),
            "grid_label": label,
            "branch_score_offsets_json": json.dumps(branch_offsets, sort_keys=True),
            "source_score_offsets_json": json.dumps(source_offsets, sort_keys=True),
        }
        summary |= build_reservoir_summary(rows, reservoir)
        if truth_rows is not None:
            frame_rows, pooled, by_sequence = build_oracle_recall_tables(
                reservoir,
                truth_rows,
                top_k_values=top_k_tuple,
                max_truth_time_delta_s=float(max_truth_time_delta_s),
            )
            if not pooled.empty:
                summary |= pooled.iloc[0].to_dict()
            frame_tables[label] = frame_rows
            by_sequence_tables[label] = by_sequence
        summary_records.append(summary)
        reservoirs[label] = reservoir
    summary_frame = pd.DataFrame.from_records(summary_records)
    summary_frame = _sort_summary(summary_frame, selection_metric=selection_metric)
    best_reservoir: pd.DataFrame | None = None
    if not summary_frame.empty:
        best_label = str(summary_frame.iloc[0]["grid_label"])
        best_reservoir = reservoirs.get(best_label)
    if output_dir is not None:
        _write_outputs(
            output_dir=Path(output_dir),
            summary_frame=summary_frame,
            reservoirs=reservoirs,
            frame_tables=frame_tables,
            by_sequence_tables=by_sequence_tables,
            best_reservoir=best_reservoir if write_best_reservoir else None,
        )
    return summary_frame, best_reservoir if write_best_reservoir else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-reservoir-grid",
        description="sweep branch/source score offsets for MMUAD candidate reservoirs",
    )
    parser.add_argument("--candidate", action="append", default=[], help="candidate CSV as BRANCH=path")
    parser.add_argument("--candidate-csv", action="append", default=[], help="alias for --candidate")
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--branch-score-offset-grid",
        action="append",
        default=[],
        help="branch=value[,value...] additive grid; may be repeated",
    )
    parser.add_argument(
        "--source-score-offset-grid",
        action="append",
        default=[],
        help="source=value[,value...] additive grid; may be repeated",
    )
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
    if candidates.empty:
        raise ValueError("at least one non-empty --candidate BRANCH=PATH CSV is required")
    truth = None if args.truth_csv is None else pd.read_csv(args.truth_csv)
    summary, best_reservoir = run_candidate_reservoir_offset_grid(
        candidates,
        truth=truth,
        branch_offset_grid=args.branch_score_offset_grid,
        source_offset_grid=args.source_score_offset_grid,
        output_dir=args.output_dir,
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
    print("mmuad_candidate_reservoir_grid=ok")
    print(f"summary_csv={args.output_dir / 'mmuad_candidate_reservoir_offset_grid_summary.csv'}")
    if not summary.empty:
        print(f"best_grid_label={summary.iloc[0]['grid_label']}")
        if args.selection_metric in summary.columns:
            print(f"best_{args.selection_metric}={summary.iloc[0][args.selection_metric]}")
    if best_reservoir is not None:
        print(f"best_reservoir_rows={len(best_reservoir)}")
    return 0


def _with_adjusted_scores(
    candidates: pd.DataFrame,
    *,
    branch_offsets: dict[str, float],
    source_offsets: dict[str, float],
    score_column: str,
    fallback_score_column: str,
) -> pd.DataFrame:
    rows = candidates.copy()
    base = _numeric_column(rows, score_column, default=np.nan)
    fallback = _numeric_column(rows, fallback_score_column, default=1.0)
    rows["candidate_reservoir_grid_base_score"] = base.fillna(fallback).fillna(0.0)
    rows["candidate_reservoir_grid_branch_offset"] = (
        rows.get("candidate_branch", "").astype(str).map(branch_offsets).fillna(0.0).astype(float)
    )
    rows["candidate_reservoir_grid_source_offset"] = (
        rows.get("source", "").astype(str).map(source_offsets).fillna(0.0).astype(float)
    )
    rows["candidate_reservoir_grid_score"] = (
        rows["candidate_reservoir_grid_base_score"]
        + rows["candidate_reservoir_grid_branch_offset"]
        + rows["candidate_reservoir_grid_source_offset"]
    )
    return rows


def _offset_config_grid(
    branch_specs: Sequence[str],
    source_specs: Sequence[str],
) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    branch_items = _parse_offset_specs(branch_specs)
    source_items = _parse_offset_specs(source_specs)
    if not branch_items:
        branch_items = [("__none__", (0.0,))]
    if not source_items:
        source_items = [("__none__", (0.0,))]
    branch_names = [name for name, _ in branch_items]
    source_names = [name for name, _ in source_items]
    configs: list[tuple[str, dict[str, float], dict[str, float]]] = []
    for branch_values in product(*[values for _, values in branch_items]):
        branch_offsets = {
            name: float(value)
            for name, value in zip(branch_names, branch_values, strict=True)
            if name != "__none__" and float(value) != 0.0
        }
        for source_values in product(*[values for _, values in source_items]):
            source_offsets = {
                name: float(value)
                for name, value in zip(source_names, source_values, strict=True)
                if name != "__none__" and float(value) != 0.0
            }
            label = _offset_label(branch_offsets, source_offsets)
            configs.append((label, branch_offsets, source_offsets))
    return configs


def _parse_offset_specs(specs: Sequence[str]) -> list[tuple[str, tuple[float, ...]]]:
    parsed: list[tuple[str, tuple[float, ...]]] = []
    for spec in specs:
        if "=" not in str(spec):
            raise ValueError(f"offset grid spec must be NAME=v1,v2,...; got {spec!r}")
        name, values_text = str(spec).split("=", 1)
        name = name.strip()
        values = tuple(float(value) for value in values_text.split(",") if value.strip())
        if not name or not values:
            raise ValueError(f"invalid offset grid spec {spec!r}")
        parsed.append((name, values))
    return parsed


def _offset_label(branch_offsets: dict[str, float], source_offsets: dict[str, float]) -> str:
    parts = []
    for prefix, offsets in (("branch", branch_offsets), ("source", source_offsets)):
        for name, value in sorted(offsets.items()):
            parts.append(f"{prefix}_{_sanitize_label(name)}_{_format_float(value)}")
    return "identity" if not parts else "__".join(parts)


def _sanitize_label(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_").replace("=", "_")


def _format_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _numeric_column(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(default, index=rows.index, dtype=float)


def _sort_summary(summary: pd.DataFrame, *, selection_metric: str) -> pd.DataFrame:
    if summary.empty:
        return summary
    if selection_metric in summary.columns:
        values = pd.to_numeric(summary[selection_metric], errors="coerce")
        return (
            summary.assign(_sort_metric=values)
            .sort_values(["_sort_metric", "grid_label"], na_position="last")
            .drop(columns=["_sort_metric"])
            .reset_index(drop=True)
        )
    return summary.sort_values("grid_label").reset_index(drop=True)


def _write_outputs(
    *,
    output_dir: Path,
    summary_frame: pd.DataFrame,
    reservoirs: dict[str, pd.DataFrame],
    frame_tables: dict[str, pd.DataFrame],
    by_sequence_tables: dict[str, pd.DataFrame],
    best_reservoir: pd.DataFrame | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "mmuad_candidate_reservoir_offset_grid_summary.csv"
    summary_json = output_dir / "mmuad_candidate_reservoir_offset_grid_summary.json"
    summary_frame.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary_frame.to_dict(orient="records"), indent=2), encoding="utf-8")
    if best_reservoir is not None:
        best_reservoir.to_csv(output_dir / "best_candidate_reservoir.csv", index=False)
    for label, frame_rows in frame_tables.items():
        frame_rows.to_csv(output_dir / f"oracle_frames_{_sanitize_label(label)}.csv", index=False)
    for label, by_sequence in by_sequence_tables.items():
        by_sequence.to_csv(output_dir / f"oracle_by_sequence_{_sanitize_label(label)}.csv", index=False)
    if reservoirs and len(reservoirs) <= 20:
        reservoir_dir = output_dir / "reservoirs"
        reservoir_dir.mkdir(exist_ok=True)
        for label, reservoir in reservoirs.items():
            reservoir.to_csv(reservoir_dir / f"reservoir_{_sanitize_label(label)}.csv", index=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
