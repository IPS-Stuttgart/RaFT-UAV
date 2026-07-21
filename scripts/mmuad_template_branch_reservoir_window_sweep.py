#!/usr/bin/env python
"""Sweep template-aligned MMUAD branch reservoirs over time windows.

The Codabench/UG2+ Track 5 test path is defined by an official
Sequence/Timestamp template.  This helper is truth-free: it builds one
branch-preserving reservoir per requested candidate/template time window and
reports coverage, fallback usage, branch diversity, and retained-row counts.
That makes it easier to choose a safe hidden-test candidate window before
running expensive tracker or mixture-MAP inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from mmuad_template_branch_reservoir import (  # noqa: E402
    BRANCH_SUMMARY_CSV,
    FRAME_SUMMARY_CSV,
    PROVENANCE_JSON,
    RESERVOIR_CSV,
    SCORE_NORMALIZATION_CHOICES,
    build_template_branch_reservoir,
    load_branch_candidate_inputs,
    load_official_track5_template_file,
    parse_candidate_input,
)

SUMMARY_CSV = "mmuad_template_branch_reservoir_window_sweep_summary.csv"
SUMMARY_JSON = "mmuad_template_branch_reservoir_window_sweep_summary.json"


def run_template_window_sweep(
    candidates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    output_dir: Path,
    max_time_delta_s_values: Iterable[float],
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    global_top_n: int = 20,
    score_column: str = "ranker_score",
    score_floor_quantile: float | None = None,
    max_candidates_per_template: int | None = None,
    score_normalization: str = "none",
    min_candidates_per_template: int = 0,
    fallback_max_time_delta_s: float | None = None,
) -> pd.DataFrame:
    """Build reservoirs for each time window and write a compact sweep summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for max_time_delta_s in _parse_float_values(max_time_delta_s_values):
        label = _window_label(max_time_delta_s)
        variant_dir = output_dir / label
        variant_dir.mkdir(parents=True, exist_ok=True)
        reservoir, frame_summary, branch_summary = build_template_branch_reservoir(
            candidates,
            template,
            max_time_delta_s=float(max_time_delta_s),
            per_source_top_n=per_source_top_n,
            per_branch_top_n=per_branch_top_n,
            global_top_n=global_top_n,
            score_column=score_column,
            score_floor_quantile=score_floor_quantile,
            max_candidates_per_template=max_candidates_per_template,
            score_normalization=score_normalization,
            min_candidates_per_template=min_candidates_per_template,
            fallback_max_time_delta_s=fallback_max_time_delta_s,
        )
        reservoir_path = variant_dir / RESERVOIR_CSV
        frame_summary_path = variant_dir / FRAME_SUMMARY_CSV
        branch_summary_path = variant_dir / BRANCH_SUMMARY_CSV
        provenance_path = variant_dir / PROVENANCE_JSON
        reservoir.to_csv(reservoir_path, index=False)
        frame_summary.to_csv(frame_summary_path, index=False)
        branch_summary.to_csv(branch_summary_path, index=False)
        provenance = {
            "schema": "raft-uav-mmuad-template-branch-reservoir-window-sweep-v1",
            "max_time_delta_s": float(max_time_delta_s),
            "per_source_top_n": int(per_source_top_n),
            "per_branch_top_n": int(per_branch_top_n),
            "global_top_n": int(global_top_n),
            "score_column": str(score_column),
            "score_floor_quantile": score_floor_quantile,
            "max_candidates_per_template": max_candidates_per_template,
            "score_normalization": str(score_normalization),
            "min_candidates_per_template": int(min_candidates_per_template),
            "fallback_max_time_delta_s": fallback_max_time_delta_s,
            "reservoir_csv": str(reservoir_path),
            "frame_summary_csv": str(frame_summary_path),
            "branch_summary_csv": str(branch_summary_path),
        }
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        records.append(
            _summary_record(
                max_time_delta_s=max_time_delta_s,
                label=label,
                reservoir=reservoir,
                frame_summary=frame_summary,
                branch_summary=branch_summary,
                reservoir_path=reservoir_path,
                frame_summary_path=frame_summary_path,
                branch_summary_path=branch_summary_path,
                provenance_path=provenance_path,
            )
        )
    summary = pd.DataFrame.from_records(records)
    summary.to_csv(output_dir / SUMMARY_CSV, index=False)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps({"rows": summary.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-csv", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="BRANCH=PATH",
        help="candidate CSV to include; may be repeated; plain PATH uses file stem as branch",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-time-delta-s",
        action="append",
        default=None,
        help=(
            "candidate/template window in seconds; may be repeated or comma separated; "
            "defaults to 0.25,0.5,1.0 when omitted"
        ),
    )
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--max-candidates-per-template", type=int)
    parser.add_argument(
        "--score-normalization",
        choices=SCORE_NORMALIZATION_CHOICES,
        default="none",
    )
    parser.add_argument("--min-candidates-per-template", type=int, default=0)
    parser.add_argument("--fallback-max-time-delta-s", type=float)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    try:
        max_time_delta_s_values = _parse_float_values(
            args.max_time_delta_s or ("0.25,0.5,1.0",)
        )
    except ValueError as exc:
        parser.error(str(exc))
    inputs = tuple(parse_candidate_input(value) for value in args.candidate_csv)
    candidates = load_branch_candidate_inputs(inputs)
    template = load_official_track5_template_file(args.template_csv)
    summary = run_template_window_sweep(
        candidates,
        template,
        output_dir=args.output_dir,
        max_time_delta_s_values=max_time_delta_s_values,
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
        max_candidates_per_template=args.max_candidates_per_template,
        score_normalization=str(args.score_normalization),
        min_candidates_per_template=int(args.min_candidates_per_template),
        fallback_max_time_delta_s=args.fallback_max_time_delta_s,
    )
    print("mmuad_template_branch_reservoir_window_sweep=ok")
    print(f"summary_csv={args.output_dir / SUMMARY_CSV}")
    print(f"variant_count={len(summary)}")
    return 0


def _summary_record(
    *,
    max_time_delta_s: float,
    label: str,
    reservoir: pd.DataFrame,
    frame_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    reservoir_path: Path,
    frame_summary_path: Path,
    branch_summary_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    template_rows = int(len(frame_summary))
    reservoir_counts = _numeric(frame_summary.get("reservoir_count"))
    fallback_counts = _numeric(frame_summary.get("fallback_retained_count"))
    branch_counts = _numeric(frame_summary.get("branch_count_reservoir"))
    source_counts = _numeric(frame_summary.get("source_count_reservoir"))
    max_abs_deltas = _numeric(frame_summary.get("max_abs_time_delta_s"))
    with_candidates = int((reservoir_counts > 0).sum()) if len(reservoir_counts) else 0
    missing = template_rows - with_candidates
    return {
        "max_time_delta_s": float(max_time_delta_s),
        "window_label": label,
        "template_rows": template_rows,
        "templates_with_candidates": with_candidates,
        "templates_missing_candidates": int(missing),
        "template_candidate_coverage_fraction": float(with_candidates / template_rows)
        if template_rows
        else 0.0,
        "reservoir_rows": int(len(reservoir)),
        "branch_summary_rows": int(len(branch_summary)),
        "mean_reservoir_count": _safe_mean(reservoir_counts),
        "p95_reservoir_count": _safe_percentile(reservoir_counts, 95),
        "mean_branch_count_reservoir": _safe_mean(branch_counts),
        "mean_source_count_reservoir": _safe_mean(source_counts),
        "fallback_rows": int(fallback_counts.sum()) if len(fallback_counts) else 0,
        "p95_max_abs_time_delta_s": _safe_percentile(max_abs_deltas, 95),
        "reservoir_csv": str(reservoir_path),
        "frame_summary_csv": str(frame_summary_path),
        "branch_summary_csv": str(branch_summary_path),
        "provenance_json": str(provenance_path),
    }


def _parse_float_values(values: Iterable[float | str]) -> tuple[float, ...]:
    parsed: list[float] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                parsed_value = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_time_delta_s values must be finite, non-negative numbers; "
                    f"got {item!r}"
                ) from exc
            if not np.isfinite(parsed_value) or parsed_value < 0.0:
                raise ValueError(
                    "max_time_delta_s values must be finite, non-negative numbers; "
                    f"got {item!r}"
                )
            if parsed_value == 0.0:
                parsed_value = 0.0
            parsed.append(parsed_value)
    if not parsed:
        raise ValueError("provide at least one max_time_delta_s value")
    return tuple(sorted(set(parsed)))


def _window_label(value: float) -> str:
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(
            "max_time_delta_s values must be finite, non-negative numbers; "
            f"got {value!r}"
        )
    if numeric_value == 0.0:
        numeric_value = 0.0
    text = (
        repr(numeric_value)
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
    )
    return f"window_{text}s"


def _numeric(values: object) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    series = pd.to_numeric(values, errors="coerce")
    array = np.asarray(series, dtype=float)
    return array[np.isfinite(array)]


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else np.nan


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
