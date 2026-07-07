"""One-shot MMUAD candidate-assignment diagnostic report.

The lower-level diagnostics are intentionally small and composable. This module
runs the common bundle for reservoir/mixture experiments: frame-level assignment
diagnostics, branch/source summaries, contiguous failure blocks, and a ranked
action plan for the next reservoir/mixture experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad.candidate_assignment_action_plan import (
    build_candidate_assignment_action_plan,
)
from raft_uav.mmuad.candidate_assignment_action_plan import (
    write_candidate_assignment_action_plan_outputs,
)
from raft_uav.mmuad.candidate_assignment_blocks import (
    build_candidate_assignment_block_tables,
)
from raft_uav.mmuad.candidate_assignment_blocks import (
    write_candidate_assignment_block_outputs,
)
from raft_uav.mmuad.candidate_assignment_branch_summary import (
    build_candidate_assignment_branch_summary,
)
from raft_uav.mmuad.candidate_assignment_branch_summary import (
    write_candidate_assignment_branch_summary,
)
from raft_uav.mmuad.candidate_assignment_diagnostics import (
    CandidateAssignmentDiagnosticsConfig,
)
from raft_uav.mmuad.candidate_assignment_diagnostics import (
    build_candidate_assignment_diagnostics,
)
from raft_uav.mmuad.candidate_assignment_diagnostics import (
    write_candidate_assignment_diagnostics,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file

REPORT_JSON = "mmuad_candidate_assignment_report.json"


def run_candidate_assignment_report(
    *,
    assignments: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    config: CandidateAssignmentDiagnosticsConfig | None = None,
    block_max_gap_s: float = 1.0,
    action_top_n_blocks: int = 20,
    action_duration_weight: float = 1.0,
    action_frame_weight: float = 1.0,
    action_error_weight: float = 1.0,
    action_regret_weight: float = 1.0,
    action_buried_weight: float = 0.5,
) -> dict[str, Any]:
    """Write frame, branch, block, and action-plan assignment diagnostics."""

    config = config or CandidateAssignmentDiagnosticsConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame_rows, summary = build_candidate_assignment_diagnostics(
        assignments,
        truth,
        config=config,
    )
    diagnostic_paths = write_candidate_assignment_diagnostics(
        frame_rows=frame_rows,
        summary=summary,
        output_dir=output,
        config=config,
    )

    branch_summary = build_candidate_assignment_branch_summary(frame_rows)
    branch_paths = write_candidate_assignment_branch_summary(
        output_dir=output,
        summary=branch_summary,
        provenance={"frame_csv": str(diagnostic_paths["frame_csv"])},
    )

    blocks, block_summary = build_candidate_assignment_block_tables(
        frame_rows,
        max_gap_s=float(block_max_gap_s),
    )
    block_paths = write_candidate_assignment_block_outputs(
        output_dir=output,
        blocks=blocks,
        summary=block_summary,
        provenance={
            "frame_csv": str(diagnostic_paths["frame_csv"]),
            "max_gap_s": float(block_max_gap_s),
        },
    )

    action_rows, action_summary = build_candidate_assignment_action_plan(
        blocks,
        top_n_blocks=int(action_top_n_blocks),
        duration_weight=float(action_duration_weight),
        frame_weight=float(action_frame_weight),
        error_weight=float(action_error_weight),
        regret_weight=float(action_regret_weight),
        buried_weight=float(action_buried_weight),
    )
    action_paths = write_candidate_assignment_action_plan_outputs(
        output_dir=output,
        action_rows=action_rows,
        action_summary=action_summary,
    )

    report_path = output / REPORT_JSON
    path_bundle = {
        **diagnostic_paths,
        **branch_paths,
        **block_paths,
        **action_paths,
        "report_json": report_path,
    }
    report = {
        "schema": "raft-uav-mmuad-candidate-assignment-report-v2",
        "config": asdict(config),
        "block_max_gap_s": float(block_max_gap_s),
        "action_plan_config": {
            "top_n_blocks": int(action_top_n_blocks),
            "duration_weight": float(action_duration_weight),
            "frame_weight": float(action_frame_weight),
            "error_weight": float(action_error_weight),
            "regret_weight": float(action_regret_weight),
            "buried_weight": float(action_buried_weight),
        },
        "frame_count": int(len(frame_rows)),
        "summary_row_count": int(len(summary)),
        "branch_summary_row_count": int(len(branch_summary)),
        "block_count": int(len(blocks)),
        "block_summary_row_count": int(len(block_summary)),
        "action_count": int(len(action_rows)),
        "action_summary_row_count": int(len(action_summary)),
        "paths": _string_paths(path_bundle),
        "pooled": _pooled_summary(summary),
        "top_action": _top_action(action_rows),
    }
    report_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-assignment-report",
        description=(
            "write MMUAD candidate-assignment frame, branch, block, and action diagnostics"
        ),
    )
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--good-candidate-threshold-m", type=float, default=5.0)
    parser.add_argument("--regret-threshold-m", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--block-max-gap-s", type=float, default=1.0)
    parser.add_argument("--action-top-n-blocks", type=int, default=20)
    parser.add_argument("--action-duration-weight", type=float, default=1.0)
    parser.add_argument("--action-frame-weight", type=float, default=1.0)
    parser.add_argument("--action-error-weight", type=float, default=1.0)
    parser.add_argument("--action-regret-weight", type=float, default=1.0)
    parser.add_argument("--action-buried-weight", type=float, default=0.5)
    args = parser.parse_args(argv)

    config = CandidateAssignmentDiagnosticsConfig(
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
        good_candidate_threshold_m=float(args.good_candidate_threshold_m),
        regret_threshold_m=float(args.regret_threshold_m),
        top_k=int(args.top_k),
    )
    report = run_candidate_assignment_report(
        assignments=pd.read_csv(args.assignments_csv),
        truth=load_evaluation_truth_file(args.truth_csv).rows,
        output_dir=args.output_dir,
        config=config,
        block_max_gap_s=float(args.block_max_gap_s),
        action_top_n_blocks=int(args.action_top_n_blocks),
        action_duration_weight=float(args.action_duration_weight),
        action_frame_weight=float(args.action_frame_weight),
        action_error_weight=float(args.action_error_weight),
        action_regret_weight=float(args.action_regret_weight),
        action_buried_weight=float(args.action_buried_weight),
    )
    print("mmuad_candidate_assignment_report=ok")
    print(f"frame_count={report['frame_count']}")
    print(f"block_count={report['block_count']}")
    print(f"action_count={report['action_count']}")
    pooled = report.get("pooled", {})
    if pooled.get("state_error_3d_m_mse") is not None:
        print(f"state_error_3d_m_mse={pooled['state_error_3d_m_mse']}")
    top_action = report.get("top_action", {})
    if top_action.get("recommended_action"):
        print(f"top_recommended_action={top_action['recommended_action']}")
    for key, value in report["paths"].items():
        print(f"{key}={value}")
    return 0


def _pooled_summary(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {}
    mask = (summary["sequence_id"] == "__pooled__") & (
        summary["assignment_failure_mode"] == "__all__"
    )
    if not mask.any():
        return {}
    return _jsonable(summary.loc[mask].iloc[0].to_dict())


def _top_action(action_rows: pd.DataFrame) -> dict[str, Any]:
    if action_rows.empty:
        return {}
    return _jsonable(action_rows.iloc[0].to_dict())


def _string_paths(paths: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in paths.items()}


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
