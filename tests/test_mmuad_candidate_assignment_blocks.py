from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_assignment_blocks import (
    build_candidate_assignment_block_tables,
    main as assignment_blocks_main,
)


def _frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 5 + ["seqB"] * 2,
            "time_s": [0.0, 0.5, 1.0, 5.0, 5.5, 0.0, 0.5],
            "assignment_failure_mode": [
                "good_candidate_buried",
                "good_candidate_buried",
                "covered",
                "missing_good_candidate_in_assignments",
                "missing_good_candidate_in_assignments",
                "covered",
                "covered",
            ],
            "state_error_3d_m": [9.0, 8.0, 0.5, 12.0, 14.0, 0.4, 0.5],
            "oracle_error_3d_m": [0.4, 0.6, 0.5, 10.0, 11.0, 0.4, 0.5],
            "state_regret_m": [8.6, 7.4, 0.0, 2.0, 3.0, 0.0, 0.0],
            "dominant_regret_m": [9.6, 8.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            "weighted_regret_m": [7.0, 6.0, 0.0, 1.0, 2.0, 0.0, 0.0],
            "oracle_weight_rank": [5, 6, 1, 1, 1, 1, 1],
            "oracle_candidate_branch": ["raw", "raw", "raw", "dynamic", "dynamic", "raw", "raw"],
            "dominant_candidate_branch": [
                "translated",
                "translated",
                "raw",
                "dynamic",
                "dynamic",
                "raw",
                "raw",
            ],
            "oracle_source": ["lidar", "lidar", "lidar", "livox", "livox", "lidar", "lidar"],
            "dominant_source": ["livox", "livox", "lidar", "livox", "livox", "lidar", "lidar"],
            "assignment_entropy": [0.5, 0.6, 0.0, 0.1, 0.2, 0.0, 0.0],
            "oracle_in_topk_by_weight": [False, False, True, True, True, True, True],
            "dominant_is_oracle": [False, False, True, True, True, True, True],
        }
    )


def test_assignment_blocks_split_modes_and_summarize() -> None:
    blocks, summary = build_candidate_assignment_block_tables(_frame_rows(), max_gap_s=1.0)

    assert set(blocks["assignment_failure_mode"]) == {
        "covered",
        "good_candidate_buried",
        "missing_good_candidate_in_assignments",
    }
    buried = blocks.loc[blocks["assignment_failure_mode"] == "good_candidate_buried"].iloc[0]
    assert buried["frame_count"] == 2
    assert buried["dominant_oracle_branch"] == "raw"
    assert buried["dominant_mixture_branch"] == "translated"
    assert buried["state_regret_m_max"] > 8.0
    missing = blocks.loc[
        blocks["assignment_failure_mode"] == "missing_good_candidate_in_assignments"
    ].iloc[0]
    assert missing["frame_count"] == 2
    pooled_modes = set(summary.loc[summary["sequence_id"] == "__pooled__", "assignment_failure_mode"])
    assert "good_candidate_buried" in pooled_modes
    assert "missing_good_candidate_in_assignments" in pooled_modes


def test_assignment_blocks_cli_writes_artifacts(tmp_path: Path) -> None:
    frame_csv = tmp_path / "frames.csv"
    output_dir = tmp_path / "out"
    _frame_rows().to_csv(frame_csv, index=False)

    status = assignment_blocks_main(
        [
            "--frame-csv",
            str(frame_csv),
            "--output-dir",
            str(output_dir),
            "--max-gap-s",
            "1",
        ]
    )

    assert status == 0
    blocks_csv = output_dir / "mmuad_candidate_assignment_blocks.csv"
    summary_csv = output_dir / "mmuad_candidate_assignment_block_summary.csv"
    summary_json = output_dir / "mmuad_candidate_assignment_block_summary.json"
    assert blocks_csv.exists()
    assert summary_csv.exists()
    assert summary_json.exists()
    blocks = pd.read_csv(blocks_csv)
    assert "state_regret_m_max" in blocks.columns
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["block_count"] == len(blocks)
    assert payload["max_gap_s"] == 1.0
