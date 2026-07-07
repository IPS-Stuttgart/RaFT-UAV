from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_assignment_action_plan import (
    build_candidate_assignment_action_plan,
)
from raft_uav.mmuad.candidate_assignment_action_plan import main as action_plan_main


def _block_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "sequence_id": "seqA",
                "block_id": 1,
                "assignment_failure_mode": "good_candidate_buried",
                "duration_s": 12.0,
                "frame_count": 30,
                "state_error_3d_m_max": 40.0,
                "state_regret_m_p95": 35.0,
                "oracle_in_topk_by_weight_rate": 0.1,
                "dominant_matches_oracle_rate": 0.2,
            },
            {
                "sequence_id": "seqA",
                "block_id": 2,
                "assignment_failure_mode": "covered",
                "duration_s": 3.0,
                "frame_count": 5,
                "state_error_3d_m_max": 4.0,
                "state_regret_m_p95": 1.0,
                "oracle_in_topk_by_weight_rate": 1.0,
                "dominant_matches_oracle_rate": 1.0,
            },
            {
                "sequence_id": "seqB",
                "block_id": 1,
                "assignment_failure_mode": "smoothing_assignment_gap",
                "duration_s": 8.0,
                "frame_count": 12,
                "state_error_3d_m_max": 25.0,
                "state_regret_m_p95": 18.0,
                "oracle_in_topk_by_weight_rate": 1.0,
                "dominant_matches_oracle_rate": 0.9,
            },
        ],
    )


def test_assignment_action_plan_prioritizes_buried_oracle_blocks() -> None:
    actions, summary = build_candidate_assignment_action_plan(_block_rows(), top_n_blocks=2)

    assert actions.iloc[0]["recommended_action"] == "increase_reservoir_recall_or_balance"
    assert actions.iloc[0]["action_rank"] == 1
    summary_actions = set(summary["recommended_action"].astype(str))
    assert "increase_reservoir_recall_or_balance" in summary_actions
    assert "retune_mixture_smoother" in summary_actions


def test_assignment_action_plan_cli_writes_outputs(tmp_path: Path) -> None:
    blocks_csv = tmp_path / "blocks.csv"
    output_dir = tmp_path / "out"
    _block_rows().to_csv(blocks_csv, index=False)

    status = action_plan_main(
        [
            "--blocks-csv",
            str(blocks_csv),
            "--output-dir",
            str(output_dir),
            "--top-n-blocks",
            "2",
        ]
    )

    assert status == 0
    rows_csv = output_dir / "mmuad_candidate_assignment_action_plan.csv"
    summary_csv = output_dir / "mmuad_candidate_assignment_action_summary.csv"
    summary_json = output_dir / "mmuad_candidate_assignment_action_summary.json"
    assert rows_csv.exists()
    assert summary_csv.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "raft-uav-mmuad-candidate-assignment-action-plan-v1"
    assert payload["action_count"] == 2
    rows = pd.read_csv(rows_csv)
    assert rows.iloc[0]["recommended_action"] == "increase_reservoir_recall_or_balance"
