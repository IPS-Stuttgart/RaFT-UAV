#!/usr/bin/env python
"""Sweep truth-free MMUAD template-reservoir settings.

The CVPR UG2+/MMUAD Track 5 hidden-test workflow must align predictions to an
official Sequence/Timestamp template.  The template branch-reservoir builder is
truth-free and inference-safe, but choosing a time window or score normalization
blindly can either lose candidates or create very large candidate sets.  This
helper runs a compact sweep and summarizes coverage/retention so the next
tracker or mixture-MAP run can use a disciplined reservoir variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from mmuad_build_branch_reservoir import (  # noqa: E402
    load_branch_candidate_inputs,
    parse_candidate_input,
)
from mmuad_template_branch_reservoir import (  # noqa: E402
    BRANCH_SUMMARY_CSV,
    FRAME_SUMMARY_CSV,
    PROVENANCE_JSON,
    RESERVOIR_CSV,
    write_template_branch_reservoir_artifacts,
)
from raft_uav.mmuad.submission import load_official_track5_template_file  # noqa: E402

SUMMARY_CSV = "mmuad_template_reservoir_window_sweep_summary.csv"
SUMMARY_JSON = "mmuad_template_reservoir_window_sweep_summary.json"


def run_template_reservoir_window_sweep(
    candidates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    output_dir: Path,
    max_time_delta_s_values: Iterable[float],
    score_normalization_values: Iterable[str] = ("none",),
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    global_top_n: int = 20,
    score_column: str = "ranker_score",
    score_floor_quantile: float | None = None,
    max_candidates_per_template: int | None = None,
    min_candidates_per_template: int = 0,
    fallback_max_time_delta_s: float | None = None,
) -> pd.DataFrame:
    """Build one template-aligned reservoir per sweep setting."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for max_time_delta_s in _parse_float_values(max_time_delta_s_values):
        for score_normalization in _parse_string_values(score_normalization_values):
            label = _variant_label(max_time_delta_s, score_normalization)
            variant_dir = output_dir / label
            paths = write_template_branch_reservoir_artifacts(
                candidates,
                template,
                output_dir=variant_dir,
                max_time_delta_s=float(max_time_delta_s),
                per_source_top_n=int(per_source_top_n),
                per_branch_top_n=int(per_branch_top_n),
                global_top_n=int(global_top_n),
                score_column=str(score_column),
                score_floor_quantile=score_floor_quantile,
                max_candidates_per_template=max_candidates_per_template,
                score_normalization=str(score_normalization),
                min_candidates_per_template=int(min_candidates_per_template),
                fallback_max_time_delta_s=fallback_max_time_delta_s,
                provenance={
                    "sweep_variant_label": label,
                    "sweep_output_dir": str(output_dir),
                },
            )
            frame_summary = pd.read_csv(paths["frame_summary_csv"])
            branch_summary = pd.read_csv(paths["branch_summary_csv"])
            reservoir = pd.read_csv(paths["reservoir_csv"])
            records.append(
                _variant_summary_record(
                    label=label,
                    max_time_delta_s=float(max_time_delta_s),
                    score_normalization=str(score_normalization),
                    frame_summary=frame_summary,
                    branch_summary=branch_summary,
                    reservoir=reservoir,
                    paths=paths,
                )
            )
    summary = pd.DataFrame.from_records(records)
    summary = _sort_summary(summary)
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
    parser.add_argument("--max-time-delta-s", action="append", default=["0.5"])
    parser.add_argument("--score-normalization", action="append", default=["none"])
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--max-candidates-per-template", type=int)
    parser.add_argument("--min-candidates-per-template", type=int, default=0)
    parser.add_argument("--fallback-max-time-delta-s", type=float)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    inputs = tuple(parse_candidate_input(value) for value in args.candidate_csv)
    candidates = load_branch_candidate_inputs(inputs)
    template = load_official_track5_template_file(args.template_csv)
    summary = run_template_reservoir_window_sweep(
        candidates,
        template,
        output_dir=args.output_dir,
        max_time_delta_s_values=_parse_float_values(args.max_time_delta_s),
        score_normalization_values=_parse_string_values(args.score_normalization),
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
        max_candidates_per_template=args.max_candidates_per_template,
        min_candidates_per_template=int(args.min_candidates_per_template),
        fallback_max_time_delta_s=args.fallback_max_time_delta_s,
    )
    print("mmuad_template_reservoir_window_sweep=ok")
    print(f"summary_csv={args.output_dir / SUMMARY_CSV}")
    print(f"variant_count={len(summary)}")
    return 0


def _variant_summary_record(
    *,
    label: str,
    max_time_delta_s: float,
    score_normalization: str,
    frame_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    reservoir: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, object]:
    template_rows = int(len(frame_summary))
    covered = frame_summary.loc[pd.to_numeric(frame_summary["reservoir_count"], errors="coerce") > 0]
    reservoir_counts = pd.to_numeric(frame_summary["reservoir_count"], errors="coerce")
    fallback_count = (
        int(pd.to_numeric(frame_summary.get("fallback_retained_count", 0), errors="coerce").sum())
        if not frame_summary.empty
        else 0
    )
    return {
        "variant_label": label,
        "max_time_delta_s": float(max_time_delta_s),
        "score_normalization": score_normalization,
        "template_rows": template_rows,
        "covered_template_rows": int(len(covered)),
        "coverage_fraction": float(len(covered) / template_rows) if template_rows else 0.0,
        "missing_template_rows": int(template_rows - len(covered)),
        "reservoir_rows": int(len(reservoir)),
        "mean_reservoir_count": float(reservoir_counts.mean()) if template_rows else np.nan,
        "p95_reservoir_count": float(np.nanpercentile(reservoir_counts, 95))
        if template_rows
        else np.nan,
        "max_reservoir_count": int(reservoir_counts.max()) if template_rows else 0,
        "fallback_retained_count": fallback_count,
        "unique_branches_retained": int(reservoir["candidate_branch"].nunique())
        if not reservoir.empty and "candidate_branch" in reservoir
        else 0,
        "unique_sources_retained": int(reservoir["source"].nunique())
        if not reservoir.empty and "source" in reservoir
        else 0,
        "branch_summary_rows": int(len(branch_summary)),
        "reservoir_csv": str(paths["reservoir_csv"]),
        "frame_summary_csv": str(paths["frame_summary_csv"]),
        "branch_summary_csv": str(paths["branch_summary_csv"]),
        "provenance_json": str(paths["provenance_json"]),
    }


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return summary.sort_values(
        ["missing_template_rows", "mean_reservoir_count", "max_time_delta_s", "score_normalization"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def _parse_float_values(values: Iterable[float | str]) -> tuple[float, ...]:
    parsed: list[float] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                parsed.append(float(item))
    return tuple(sorted({max(float(value), 0.0) for value in parsed}))


def _parse_string_values(values: Iterable[str]) -> tuple[str, ...]:
    parsed: list[str] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return tuple(dict.fromkeys(parsed or ["none"]))


def _variant_label(max_time_delta_s: float, score_normalization: str) -> str:
    delta = f"{float(max_time_delta_s):.6g}".replace("-", "m").replace(".", "p")
    score = str(score_normalization).replace("_", "-").replace(" ", "-")
    return f"dt{delta}s_score-{score}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
