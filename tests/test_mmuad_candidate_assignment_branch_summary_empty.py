from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_assignment_branch_summary import (
    build_candidate_assignment_branch_summary,
)
from raft_uav.mmuad.candidate_assignment_branch_summary import main as branch_summary_main


def test_empty_branch_summary_preserves_output_schema() -> None:
    summary = build_candidate_assignment_branch_summary(
        pd.DataFrame(columns=["sequence_id"])
    )

    assert summary.empty
    assert summary.columns.tolist() == [
        "sequence_id",
        "group_label",
        "frame_count",
        "state_error_3d_m_mse",
        "oracle_error_3d_m_mse",
        "dominant_error_3d_m_mse",
        "state_vs_oracle_mse_gap",
        "dominant_vs_oracle_mse_gap",
        "state_regret_m_mean",
        "dominant_regret_m_mean",
        "state_error_3d_m_p95",
        "oracle_mixture_weight_mean",
        "oracle_weight_deficit_mean",
        "oracle_weight_rank_p50",
        "candidate_count_mean",
        "dominant_matches_oracle_rate",
        "oracle_in_topk_by_weight_rate",
        "assignment_priority_score",
        "assignment_failure_mode",
        "oracle_candidate_branch",
        "dominant_candidate_branch",
        "oracle_source",
        "dominant_source",
    ]


def test_branch_summary_cli_writes_parseable_header_only_artifacts(
    tmp_path: Path,
) -> None:
    frame_csv = tmp_path / "frames.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(columns=["sequence_id"]).to_csv(frame_csv, index=False)

    status = branch_summary_main(
        [
            "--frame-csv",
            str(frame_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    summary_csv = output_dir / "mmuad_candidate_assignment_branch_summary.csv"
    summary_json = output_dir / "mmuad_candidate_assignment_branch_summary.json"
    written = pd.read_csv(summary_csv)
    assert written.empty
    assert "assignment_priority_score" in written.columns
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["row_count"] == 0
    assert payload["summary"] == []
