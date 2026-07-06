from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_oracle_action_plan import (
    build_candidate_oracle_action_plan,
    main as action_plan_main,
)


def _blocks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB"],
            "block_id": [0, 1, 0],
            "oracle_failure_mode": [
                "good_candidate_buried",
                "missing_good_candidate",
                "covered_in_topk",
            ],
            "start_time_s": [0.0, 10.0, 0.0],
            "end_time_s": [4.0, 12.0, 1.0],
            "duration_s": [4.0, 2.0, 1.0],
            "frame_count": [9, 3, 2],
            "oracle_all_3d_m_max": [1.2, 12.0, 0.8],
            "oracle_all_rank_p95": [12.0, 3.0, 1.0],
        }
    )


def test_action_plan_prioritizes_buried_candidate_recall_blocks() -> None:
    actions, summary = build_candidate_oracle_action_plan(
        _blocks(),
        top_n_blocks=2,
        duration_weight=1.0,
        frame_weight=1.0,
        error_weight=0.1,
        rank_weight=1.0,
    )

    assert actions.iloc[0]["recommended_action"] == "improve_topk_recall_or_ranker"
    assert "reservoir" in actions.iloc[0]["recommended_method"]
    assert list(actions["action_rank"]) == [1, 2]
    assert set(summary["recommended_action"]) >= {
        "improve_topk_recall_or_ranker",
        "improve_extraction_or_calibration",
    }


def test_action_plan_cli_writes_csv_and_json(tmp_path) -> None:
    blocks_csv = tmp_path / "blocks.csv"
    output_dir = tmp_path / "out"
    _blocks().to_csv(blocks_csv, index=False)

    rc = action_plan_main(
        [
            "--blocks-csv",
            str(blocks_csv),
            "--output-dir",
            str(output_dir),
            "--top-n-blocks",
            "2",
        ]
    )

    assert rc == 0
    actions = pd.read_csv(output_dir / "mmuad_candidate_oracle_action_plan.csv")
    summary = pd.read_csv(output_dir / "mmuad_candidate_oracle_action_summary.csv")
    assert len(actions) == 2
    assert "priority_score_sum" in summary.columns
    assert (output_dir / "mmuad_candidate_oracle_action_summary.json").exists()
