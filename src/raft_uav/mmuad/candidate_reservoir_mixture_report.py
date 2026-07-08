"""One-command action report for MMUAD reservoir-mixture runs.

A reservoir-mixture run already writes estimates, retained-reservoir oracle rows,
and several local diagnostics.  This module turns one completed run directory into
an action-oriented report by joining frame-level mixture error with the retained
candidate oracle, summarizing pooled/per-sequence gaps, and classifying the
remaining bottleneck with the existing reservoir bottleneck audit.

The command is diagnostic only.  It does not require truth when the estimates CSV
already contains a 3D error column; otherwise a local validation/reference truth
CSV can be supplied to compute the mixture errors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir_bottleneck import BottleneckConfig
from raft_uav.mmuad.candidate_reservoir_bottleneck import annotate_gap_table
from raft_uav.mmuad.candidate_reservoir_bottleneck import build_bottleneck_summary
from raft_uav.mmuad.candidate_reservoir_mixture_gap_frames import build_frame_gap_table
from raft_uav.mmuad.candidate_reservoir_mixture_gap_frames import summarize_frame_gap
from raft_uav.mmuad.class_probability_csv import read_sequence_text_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file

DEFAULT_ESTIMATES_CSV = "mmuad_candidate_mixture_estimates.csv"
DEFAULT_ORACLE_FRAME_CSV = "mmuad_reservoir_mixture_oracle_frames.csv"
FRAME_GAP_CSV = "mmuad_reservoir_mixture_report_frame_gap.csv"
GAP_SUMMARY_CSV = "mmuad_reservoir_mixture_report_gap_summary.csv"
GAP_BY_SEQUENCE_CSV = "mmuad_reservoir_mixture_report_gap_by_sequence.csv"
BOTTLENECK_CSV = "mmuad_reservoir_mixture_report_bottleneck.csv"
BOTTLENECK_BY_SEQUENCE_CSV = "mmuad_reservoir_mixture_report_bottleneck_by_sequence.csv"
REPORT_JSON = "mmuad_reservoir_mixture_report.json"


def build_reservoir_mixture_report(
    *,
    estimates: pd.DataFrame,
    oracle_frames: pd.DataFrame,
    truth: pd.DataFrame | None = None,
    bottleneck_config: BottleneckConfig | None = None,
    time_round_decimals: int = 6,
    time_join_tolerance_s: float | None = None,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Build frame-gap, bottleneck, and compact JSON report objects."""

    config = bottleneck_config or BottleneckConfig()
    frame_gap = build_frame_gap_table(
        estimates,
        oracle_frames,
        truth=truth,
        time_round_decimals=int(time_round_decimals),
        time_join_tolerance_s=time_join_tolerance_s,
    )
    gap_summary = summarize_frame_gap(frame_gap)
    gap_by_sequence = summarize_frame_gap(frame_gap, group_column="sequence_id")
    bottleneck = annotate_gap_table(
        _to_bottleneck_gap_rows(gap_summary),
        config=config,
    )
    bottleneck_by_sequence = annotate_gap_table(
        _to_bottleneck_gap_rows(gap_by_sequence),
        config=config,
    )
    payload = {
        "schema": "raft-uav-mmuad-reservoir-mixture-report-v1",
        "config": {
            "bottleneck": asdict(config),
            "time_round_decimals": int(time_round_decimals),
            "time_join_tolerance_s": time_join_tolerance_s,
        },
        "frame_count": int(len(frame_gap)),
        "pooled_gap": _first_record(gap_summary),
        "pooled_bottleneck": _first_record(bottleneck),
        "bottleneck_summary": build_bottleneck_summary(
            bottleneck_by_sequence if not bottleneck_by_sequence.empty else bottleneck,
            config=config,
        ),
        "worst_sequence_bottleneck": _max_record(
            bottleneck_by_sequence,
            "assignment_gap_mse_3d_m2",
        ),
    }
    return {
        "frame_gap": frame_gap,
        "gap_summary": gap_summary,
        "gap_by_sequence": gap_by_sequence,
        "bottleneck": bottleneck,
        "bottleneck_by_sequence": bottleneck_by_sequence,
        "report": _jsonable(payload),
    }


def write_reservoir_mixture_report_outputs(
    report: dict[str, pd.DataFrame | dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Write all report artifacts and return their paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "frame_gap_csv": output / FRAME_GAP_CSV,
        "gap_summary_csv": output / GAP_SUMMARY_CSV,
        "gap_by_sequence_csv": output / GAP_BY_SEQUENCE_CSV,
        "bottleneck_csv": output / BOTTLENECK_CSV,
        "bottleneck_by_sequence_csv": output / BOTTLENECK_BY_SEQUENCE_CSV,
        "report_json": output / REPORT_JSON,
    }
    _frame(report["frame_gap"]).to_csv(paths["frame_gap_csv"], index=False)
    _frame(report["gap_summary"]).to_csv(paths["gap_summary_csv"], index=False)
    _frame(report["gap_by_sequence"]).to_csv(paths["gap_by_sequence_csv"], index=False)
    _frame(report["bottleneck"]).to_csv(paths["bottleneck_csv"], index=False)
    _frame(report["bottleneck_by_sequence"]).to_csv(
        paths["bottleneck_by_sequence_csv"],
        index=False,
    )
    paths["report_json"].write_text(
        json.dumps(_jsonable(report["report"]), indent=2),
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-reservoir-mixture-report",
        description="build an action report for a completed MMUAD reservoir-mixture run",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--estimates-csv", type=Path)
    parser.add_argument("--oracle-frame-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--target-mse-3d-m2", type=float)
    parser.add_argument("--near-oracle-ratio", type=float, default=1.15)
    parser.add_argument("--assignment-ratio", type=float, default=2.0)
    parser.add_argument("--assignment-fraction", type=float, default=0.50)
    parser.add_argument("--topk-recall-ratio", type=float, default=1.50)
    parser.add_argument("--topk-recall-absolute-gap-mse", type=float, default=10.0)
    parser.add_argument("--time-round-decimals", type=int, default=6)
    parser.add_argument(
        "--time-join-tolerance-s",
        type=float,
        help=(
            "use nearest per-sequence estimate/oracle timestamp matching within this "
            "tolerance; omit to keep exact rounded-time matching"
        ),
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else run_dir
    estimates_csv = args.estimates_csv or run_dir / DEFAULT_ESTIMATES_CSV
    oracle_frame_csv = args.oracle_frame_csv or run_dir / DEFAULT_ORACLE_FRAME_CSV
    estimates = read_sequence_text_csv(estimates_csv)
    oracle_frames = read_sequence_text_csv(oracle_frame_csv)
    truth = None if args.truth_csv is None else load_evaluation_truth_file(args.truth_csv).rows
    config = BottleneckConfig(
        near_oracle_ratio=float(args.near_oracle_ratio),
        assignment_ratio=float(args.assignment_ratio),
        assignment_fraction=float(args.assignment_fraction),
        topk_recall_ratio=float(args.topk_recall_ratio),
        topk_recall_absolute_gap_mse=float(args.topk_recall_absolute_gap_mse),
        target_mse_3d_m2=args.target_mse_3d_m2,
    )
    report = build_reservoir_mixture_report(
        estimates=estimates,
        oracle_frames=oracle_frames,
        truth=truth,
        bottleneck_config=config,
        time_round_decimals=int(args.time_round_decimals),
        time_join_tolerance_s=args.time_join_tolerance_s,
    )
    paths = write_reservoir_mixture_report_outputs(report, output_dir=output_dir)
    payload = report["report"]
    print("mmuad_reservoir_mixture_report=ok")
    print(f"frame_count={payload['frame_count']}")
    pooled = payload.get("pooled_bottleneck", {})
    if pooled.get("primary_bottleneck") is not None:
        print(f"primary_bottleneck={pooled['primary_bottleneck']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _to_bottleneck_gap_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(summary).copy()
    if rows.empty:
        return rows
    rename: dict[str, str] = {}
    for column in rows.columns:
        text = str(column)
        if text.startswith("reservoir_oracle_"):
            continue
        if text.startswith("oracle_") and text.endswith("_mse_3d_m2"):
            rename[text] = f"reservoir_{text}"
    rows = rows.rename(columns=rename)
    return rows


def _frame(value: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame.from_records([value])


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
