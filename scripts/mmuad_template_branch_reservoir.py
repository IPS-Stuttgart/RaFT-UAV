#!/usr/bin/env python
"""Build truth-free MMUAD branch reservoirs aligned to an official template.

CVPR UG2+ Track 5 submissions are scored on a fixed Sequence/Timestamp grid.
This helper keeps raw, dynamic, calibrated, or merged candidate branches around
that template grid without reading pose truth.  It is meant as the inference-safe
counterpart to oracle reservoir diagnostics before running tracker or
mixture-MAP experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from mmuad_build_branch_reservoir import (  # noqa: E402
    build_branch_summary,
    load_branch_candidate_inputs,
    parse_candidate_input,
    _finite_candidate_rows,
    _select_frame_reservoir,
)
from raft_uav.mmuad.submission import load_official_track5_template_file  # noqa: E402

RESERVOIR_CSV = "mmuad_template_branch_reservoir_candidates.csv"
FRAME_SUMMARY_CSV = "mmuad_template_branch_reservoir_frame_summary.csv"
BRANCH_SUMMARY_CSV = "mmuad_template_branch_reservoir_branch_summary.csv"
PROVENANCE_JSON = "mmuad_template_branch_reservoir_provenance.json"


def build_template_branch_reservoir(
    candidates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    global_top_n: int = 20,
    score_column: str = "ranker_score",
    score_floor_quantile: float | None = None,
    max_candidates_per_template: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a reservoir around each official template timestamp.

    The output candidate rows keep original candidate ``time_s`` and include the
    matched template timestamp plus candidate/template time offset.  The template
    supplies only Sequence/Timestamp rows; pose labels are not used.
    """

    rows = _finite_candidate_rows(candidates, score_column=score_column)
    template_rows = _normalize_template_rows(template)
    selected_frames: list[pd.DataFrame] = []
    frame_records: list[dict[str, Any]] = []

    for row_index, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        timestamp = float(template_row["timestamp_s"])
        sequence_candidates = rows.loc[rows["sequence_id"] == sequence_id]
        if sequence_candidates.empty:
            window = sequence_candidates.copy()
        else:
            deltas = pd.to_numeric(sequence_candidates["time_s"], errors="coerce") - timestamp
            window = sequence_candidates.loc[np.abs(deltas) <= float(max_time_delta_s)].copy()
            if not window.empty:
                window["template_time_delta_s"] = pd.to_numeric(window["time_s"], errors="coerce") - timestamp
        reservoir = _select_frame_reservoir(
            window,
            per_source_top_n=per_source_top_n,
            per_branch_top_n=per_branch_top_n,
            global_top_n=global_top_n,
            score_floor_quantile=score_floor_quantile,
            max_candidates_per_frame=max_candidates_per_template,
        )
        if not reservoir.empty:
            reservoir = reservoir.copy()
            reservoir["template_row_index"] = int(row_index)
            reservoir["template_sequence_id"] = sequence_id
            reservoir["template_timestamp_s"] = timestamp
            if "template_time_delta_s" not in reservoir.columns:
                reservoir["template_time_delta_s"] = pd.to_numeric(reservoir["time_s"], errors="coerce") - timestamp
            selected_frames.append(reservoir)
        frame_records.append(
            {
                "template_row_index": int(row_index),
                "sequence_id": sequence_id,
                "template_timestamp_s": timestamp,
                "candidate_count_window": int(len(window)),
                "reservoir_count": int(len(reservoir)),
                "retained_fraction": float(len(reservoir) / len(window)) if len(window) else 0.0,
                "branch_count_window": int(window["candidate_branch"].nunique(dropna=False)) if not window.empty else 0,
                "source_count_window": int(window["source"].nunique(dropna=False)) if not window.empty else 0,
                "branch_count_reservoir": int(reservoir["candidate_branch"].nunique(dropna=False)) if not reservoir.empty else 0,
                "source_count_reservoir": int(reservoir["source"].nunique(dropna=False)) if not reservoir.empty else 0,
                "min_abs_time_delta_s": float(np.nanmin(np.abs(window["template_time_delta_s"]))) if not window.empty else np.nan,
                "max_abs_time_delta_s": float(np.nanmax(np.abs(window["template_time_delta_s"]))) if not window.empty else np.nan,
            }
        )

    reservoir_rows = (
        pd.concat(selected_frames, ignore_index=True, sort=False)
        if selected_frames
        else rows.iloc[0:0].copy()
    )
    reservoir_rows = reservoir_rows.drop(
        columns=[column for column in ("_reservoir_score", "_candidate_row_id") if column in reservoir_rows.columns]
    )
    frame_summary = pd.DataFrame.from_records(frame_records)
    branch_summary = build_branch_summary(candidates, reservoir_rows, score_column=score_column)
    return reservoir_rows, frame_summary, branch_summary


def write_template_branch_reservoir_artifacts(
    candidates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    output_dir: Path,
    max_time_delta_s: float = 0.5,
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    global_top_n: int = 20,
    score_column: str = "ranker_score",
    score_floor_quantile: float | None = None,
    max_candidates_per_template: int | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write template-aligned reservoir candidates, summaries, and provenance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    reservoir, frame_summary, branch_summary = build_template_branch_reservoir(
        candidates,
        template,
        max_time_delta_s=max_time_delta_s,
        per_source_top_n=per_source_top_n,
        per_branch_top_n=per_branch_top_n,
        global_top_n=global_top_n,
        score_column=score_column,
        score_floor_quantile=score_floor_quantile,
        max_candidates_per_template=max_candidates_per_template,
    )
    paths = {
        "reservoir_csv": output_dir / RESERVOIR_CSV,
        "frame_summary_csv": output_dir / FRAME_SUMMARY_CSV,
        "branch_summary_csv": output_dir / BRANCH_SUMMARY_CSV,
        "provenance_json": output_dir / PROVENANCE_JSON,
    }
    reservoir.to_csv(paths["reservoir_csv"], index=False)
    frame_summary.to_csv(paths["frame_summary_csv"], index=False)
    branch_summary.to_csv(paths["branch_summary_csv"], index=False)
    payload = {
        "schema": "raft-uav-mmuad-template-branch-reservoir-v1",
        "max_time_delta_s": float(max_time_delta_s),
        "per_source_top_n": int(per_source_top_n),
        "per_branch_top_n": int(per_branch_top_n),
        "global_top_n": int(global_top_n),
        "score_column": str(score_column),
        "score_floor_quantile": score_floor_quantile,
        "max_candidates_per_template": max_candidates_per_template,
        "input_candidate_rows": int(len(candidates)),
        "template_rows": int(len(_normalize_template_rows(template))),
        "reservoir_rows": int(len(reservoir)),
        "provenance": provenance or {},
    }
    paths["provenance_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return paths


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
    parser.add_argument("--max-time-delta-s", type=float, default=0.5)
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
    paths = write_template_branch_reservoir_artifacts(
        candidates,
        template,
        output_dir=args.output_dir,
        max_time_delta_s=float(args.max_time_delta_s),
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
        max_candidates_per_template=args.max_candidates_per_template,
        provenance={
            "template_csv": str(args.template_csv),
            "candidate_inputs": [{"branch": item.branch, "path": str(item.path)} for item in inputs],
        },
    )
    print("mmuad_template_branch_reservoir=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    lower = {str(column).strip().lower(): column for column in template.columns}
    sequence_col = lower.get("sequence") or lower.get("sequence_id")
    timestamp_col = lower.get("timestamp") or lower.get("time_s")
    if sequence_col is None or timestamp_col is None:
        raise ValueError("template must contain Sequence/Timestamp or sequence_id/time_s columns")
    rows = pd.DataFrame(
        {
            "sequence_id": template[sequence_col].astype(str).str.strip(),
            "timestamp_s": pd.to_numeric(template[timestamp_col], errors="coerce"),
        }
    )
    finite = rows["timestamp_s"].notna() & np.isfinite(rows["timestamp_s"].to_numpy(float))
    return rows.loc[finite & rows["sequence_id"].ne("")].sort_values(
        ["sequence_id", "timestamp_s"]
    ).reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
