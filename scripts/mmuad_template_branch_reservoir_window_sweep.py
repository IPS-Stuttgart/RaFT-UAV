#!/usr/bin/env python
"""Sweep template-aligned MMUAD branch reservoirs over time windows.

The template-aligned reservoir builder is truth-free, but a single
``--max-time-delta-s`` can either miss near-synchronous branch candidates or
admit too much clutter.  This helper builds one reservoir bundle per requested
window around the official Track 5 template and writes a compact sweep summary
so downstream mixture-MAP experiments can choose a candidate-preparation bundle
by coverage/candidate-count trade-off, not by validation pose truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

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
    write_template_branch_reservoir_artifacts,
)
from raft_uav.mmuad.submission import load_official_track5_template_file  # noqa: E402

SUMMARY_CSV = "mmuad_template_branch_reservoir_window_sweep_summary.csv"
SUMMARY_JSON = "mmuad_template_branch_reservoir_window_sweep_summary.json"


def run_template_window_sweep(
    candidates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    output_dir: Path,
    max_time_delta_values_s: Iterable[float],
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    global_top_n: int = 20,
    score_column: str = "ranker_score",
    score_floor_quantile: float | None = None,
    max_candidates_per_template: int | None = None,
) -> pd.DataFrame:
    """Write one template-aligned reservoir bundle per time-window value."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for window_s in _normalize_windows(max_time_delta_values_s):
        label = _window_label(window_s)
        variant_dir = output_dir / label
        paths = write_template_branch_reservoir_artifacts(
            candidates,
            template,
            output_dir=variant_dir,
            max_time_delta_s=window_s,
            per_source_top_n=per_source_top_n,
            per_branch_top_n=per_branch_top_n,
            global_top_n=global_top_n,
            score_column=score_column,
            score_floor_quantile=score_floor_quantile,
            max_candidates_per_template=max_candidates_per_template,
            provenance={"window_s": float(window_s), "window_label": label},
        )
        frame_summary = pd.read_csv(paths["frame_summary_csv"])
        branch_summary = pd.read_csv(paths["branch_summary_csv"])
        records.append(
            {
                "max_time_delta_s": float(window_s),
                "window_label": label,
                "template_rows": int(len(frame_summary)),
                "covered_template_rows": int((frame_summary["reservoir_count"] > 0).sum())
                if not frame_summary.empty
                else 0,
                "coverage_fraction": float((frame_summary["reservoir_count"] > 0).mean())
                if not frame_summary.empty
                else 0.0,
                "mean_reservoir_count": float(frame_summary["reservoir_count"].mean())
                if not frame_summary.empty
                else 0.0,
                "p95_reservoir_count": float(frame_summary["reservoir_count"].quantile(0.95))
                if not frame_summary.empty
                else 0.0,
                "max_reservoir_count": int(frame_summary["reservoir_count"].max())
                if not frame_summary.empty
                else 0,
                "branch_summary_rows": int(len(branch_summary)),
                "reservoir_csv": str(paths["reservoir_csv"]),
                "frame_summary_csv": str(paths["frame_summary_csv"]),
                "branch_summary_csv": str(paths["branch_summary_csv"]),
                "provenance_json": str(paths["provenance_json"]),
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
        default=["0.25,0.5,1.0"],
        help="candidate/template time window in seconds; may be repeated or comma-separated",
    )
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--max-candidates-per-template", type=int)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    inputs = tuple(parse_candidate_input(value) for value in args.candidate_csv)
    candidates = load_branch_candidate_inputs(inputs)
    template = load_official_track5_template_file(args.template_csv)
    summary = run_template_window_sweep(
        candidates,
        template,
        output_dir=args.output_dir,
        max_time_delta_values_s=_parse_windows(args.max_time_delta_s),
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
        max_candidates_per_template=args.max_candidates_per_template,
    )
    print("mmuad_template_branch_reservoir_window_sweep=ok")
    print(f"summary_csv={args.output_dir / SUMMARY_CSV}")
    print(f"variant_count={len(summary)}")
    return 0


def _parse_windows(values: Iterable[str]) -> tuple[float, ...]:
    windows: list[float] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                windows.append(float(item))
    return _normalize_windows(windows)


def _normalize_windows(values: Iterable[float]) -> tuple[float, ...]:
    windows = sorted({max(float(value), 0.0) for value in values})
    return tuple(windows or [0.5])


def _window_label(window_s: float) -> str:
    text = f"{float(window_s):.6g}".replace("-", "m").replace(".", "p")
    return f"window_{text}s"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
