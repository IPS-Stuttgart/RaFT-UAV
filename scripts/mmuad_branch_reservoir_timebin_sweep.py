#!/usr/bin/env python
"""Sweep truth-free MMUAD branch reservoirs over timestamp grouping bins.

Branch candidates from static, dynamic, calibrated, or merged streams may have
small timestamp offsets.  This helper runs the existing truth-free reservoir
builder over exact timestamps and optional rounded time bins, then restores the
original candidate timestamps in the retained reservoir CSVs so they can still be
fed into downstream tracker or mixture experiments.
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
    BRANCH_SUMMARY_CSV,
    FRAME_SUMMARY_CSV,
    PROVENANCE_JSON,
    RESERVOIR_CSV,
    build_branch_summary,
    build_frame_summary,
    build_truth_free_branch_reservoir,
    load_branch_candidate_inputs,
    parse_candidate_input,
)

SUMMARY_CSV = "mmuad_branch_reservoir_timebin_sweep_summary.csv"
SUMMARY_JSON = "mmuad_branch_reservoir_timebin_sweep_summary.json"


def run_timebin_sweep(
    candidates: pd.DataFrame,
    *,
    output_dir: Path,
    time_bins_s: Iterable[float],
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    global_top_n: int = 20,
    score_column: str = "ranker_score",
    score_floor_quantile: float | None = None,
    max_candidates_per_frame: int | None = None,
) -> pd.DataFrame:
    """Build one truth-free reservoir per requested time bin."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for time_bin_s in _normalized_time_bins(time_bins_s):
        label = _time_bin_label(time_bin_s)
        variant_dir = output_dir / label
        variant_dir.mkdir(parents=True, exist_ok=True)
        selection_candidates = _candidates_for_time_bin(candidates, time_bin_s=time_bin_s)
        reservoir_for_summary = build_truth_free_branch_reservoir(
            selection_candidates,
            per_source_top_n=per_source_top_n,
            per_branch_top_n=per_branch_top_n,
            global_top_n=global_top_n,
            score_column=score_column,
            score_floor_quantile=score_floor_quantile,
            max_candidates_per_frame=max_candidates_per_frame,
        )
        reservoir_to_write = _restore_original_time(reservoir_for_summary, time_bin_s=time_bin_s)
        frame_summary = build_frame_summary(selection_candidates, reservoir_for_summary)
        branch_summary = build_branch_summary(selection_candidates, reservoir_for_summary)

        reservoir_path = variant_dir / RESERVOIR_CSV
        frame_summary_path = variant_dir / FRAME_SUMMARY_CSV
        branch_summary_path = variant_dir / BRANCH_SUMMARY_CSV
        provenance_path = variant_dir / PROVENANCE_JSON
        reservoir_to_write.to_csv(reservoir_path, index=False)
        frame_summary.to_csv(frame_summary_path, index=False)
        branch_summary.to_csv(branch_summary_path, index=False)
        provenance = {
            "time_bin_s": float(time_bin_s),
            "time_bin_label": label,
            "retained_rows": int(len(reservoir_to_write)),
            "input_rows": int(len(candidates)),
            "per_source_top_n": int(per_source_top_n),
            "per_branch_top_n": int(per_branch_top_n),
            "global_top_n": int(global_top_n),
            "score_column": str(score_column),
            "score_floor_quantile": score_floor_quantile,
            "max_candidates_per_frame": max_candidates_per_frame,
        }
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        records.append(
            {
                "time_bin_s": float(time_bin_s),
                "time_bin_label": label,
                "input_rows": int(len(candidates)),
                "reservoir_rows": int(len(reservoir_to_write)),
                "frame_rows": int(len(frame_summary)),
                "branch_rows": int(len(branch_summary)),
                "reservoir_csv": str(reservoir_path),
                "frame_summary_csv": str(frame_summary_path),
                "branch_summary_csv": str(branch_summary_path),
                "provenance_json": str(provenance_path),
            }
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
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="BRANCH=PATH",
        help="candidate CSV to include; may be repeated; plain PATH uses file stem as branch",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-bin-s", action="append", default=["0"], help="time bin in seconds; may be repeated")
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--max-candidates-per-frame", type=int)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    inputs = tuple(parse_candidate_input(value) for value in args.candidate_csv)
    candidates = load_branch_candidate_inputs(inputs)
    summary = run_timebin_sweep(
        candidates,
        output_dir=args.output_dir,
        time_bins_s=_parse_time_bins(args.time_bin_s),
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
        max_candidates_per_frame=args.max_candidates_per_frame,
    )
    print("mmuad_branch_reservoir_timebin_sweep=ok")
    print(f"summary_csv={args.output_dir / SUMMARY_CSV}")
    print(f"variant_count={len(summary)}")
    return 0


def _candidates_for_time_bin(candidates: pd.DataFrame, *, time_bin_s: float) -> pd.DataFrame:
    rows = pd.DataFrame(candidates).copy()
    rows["original_time_s"] = pd.to_numeric(rows["time_s"], errors="coerce")
    rows["reservoir_time_bin_s"] = float(time_bin_s)
    if float(time_bin_s) <= 0.0:
        rows["reservoir_group_time_s"] = rows["original_time_s"]
        return rows
    rows["reservoir_group_time_s"] = _round_time(rows["original_time_s"], float(time_bin_s))
    rows["time_s"] = rows["reservoir_group_time_s"]
    return rows


def _restore_original_time(reservoir: pd.DataFrame, *, time_bin_s: float) -> pd.DataFrame:
    rows = pd.DataFrame(reservoir).copy()
    rows["reservoir_time_bin_s"] = float(time_bin_s)
    if "original_time_s" in rows.columns:
        rows["reservoir_group_time_s"] = rows["time_s"]
        rows["time_s"] = pd.to_numeric(rows["original_time_s"], errors="coerce")
    return rows


def _round_time(times: pd.Series, time_bin_s: float) -> pd.Series:
    values = pd.to_numeric(times, errors="coerce")
    return (np.round(values / float(time_bin_s)) * float(time_bin_s)).astype(float)


def _parse_time_bins(values: Iterable[str]) -> tuple[float, ...]:
    bins: list[float] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                bins.append(float(item))
    return _normalized_time_bins(bins)


def _normalized_time_bins(values: Iterable[float]) -> tuple[float, ...]:
    unique = sorted({max(float(value), 0.0) for value in values})
    return tuple(unique or [0.0])


def _time_bin_label(time_bin_s: float) -> str:
    if float(time_bin_s) <= 0.0:
        return "timebin_exact"
    text = f"{float(time_bin_s):.6g}".replace("-", "m").replace(".", "p")
    return f"timebin_{text}s"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
