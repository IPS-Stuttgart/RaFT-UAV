from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_assignment_blocks import build_candidate_assignment_block_tables
from raft_uav.mmuad.candidate_assignment_blocks import main as assignment_blocks_main


def _frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6 + ["seqB"] * 2,
            "time_s": [0.0, 0.5, 1.0, 5.0, 5.5, 6.0, 0.0, 0.5],
            "assignment_failure_mode": [
                "covered",
                "good_candidate_buried",
                "good_candidate_buried",
                "missing_good_candidate_in_assignments",
                "missing_good_candidate_in_assignments",
                "covered",
                "smoothing_assignment_gap",
                "smoothing_assignment_gap",
            ],
            "state_error_3d_m": [0.4, 9.0, 8.0, 11.0, 12.0, 0.5, 6.0, 7.0],
            "oracle_error_3d_m": [0.2, 0.4, 0.5, 8.0, 9.0, 0.3, 0.6, 0.7],
            "state_regret_m": [0.2, 8.6, 7.5, 3.0, 3.0, 0.2, 5.4, 6.3],
            "dominant_regret_m": [0.0, 6.0, 5.0, 1.0, 1.0, 0.0, 0.5, 0.4],
            "weighted_regret_m": [0.1, 4.0, 3.5, 1.5, 1.0, 0.1, 4.0, 5.0],
            "oracle_mixture_weight": [0.8, 0.01, 0.02, 0.2, 0.3, 0.9, 0.7, 0.6],
            "oracle_weight_rank": [1, 8, 9, 2, 3, 1, 1, 1],
            "dominant_is_oracle": [True, False, False, False, False, True, True, True],
            "oracle_in_topk_by_weight": [True, False, False, True, True, True, True, True],
            "oracle_candidate_branch": ["raw", "raw", "raw", "dynamic", "dynamic", "raw", "raw", "raw"],
            "oracle_source": ["lidar", "lidar", "lidar", "livox", "livox", "lidar", "lidar", "lidar"],
            "dominant_candidate_branch": ["raw", "translated", "translated", "dynamic", "dynamic", "raw", "raw", "raw"],
            "dominant_source": ["lidar", "livox", "livox", "livox", "livox", "lidar", "lidar", "lidar"],
        }
    )


def test_assignment_blocks_group_contiguous_failure_modes() -> None:
    blocks, summary = build_candidate_assignment_block_tables(_frame_rows(), max_gap_s=1.0)

    assert set(blocks["assignment_failure_mode"]) == {
        "covered",
        "good_candidate_buried",
        "missing_good_candidate_in_assignments",
        "smoothing_assignment_gap",
    }
    buried = blocks.loc[blocks["assignment_failure_mode"] == "good_candidate_buried"].iloc[0]
    assert buried["frame_count"] == 2
    assert buried["dominant_oracle_branch"] == "raw"
    assert buried["dominant_assigned_branch"] == "translated"
    missing = blocks.loc[
        blocks["assignment_failure_mode"] == "missing_good_candidate_in_assignments"
    ].iloc[0]
    assert missing["state_error_3d_m_max"] == 12.0
    pooled_modes = set(summary.loc[summary["sequence_id"] == "__pooled__", "assignment_failure_mode"])
    assert "good_candidate_buried" in pooled_modes
    assert "missing_good_candidate_in_assignments" in pooled_modes


def test_assignment_blocks_cli_writes_artifacts(tmp_path: Path) -> None:
    frame_csv = tmp_path / "assignment_frames.csv"
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
    blocks = pd.read_csv(output_dir / "mmuad_candidate_assignment_blocks.csv")
    summary = pd.read_csv(output_dir / "mmuad_candidate_assignment_block_summary.csv")
    payload = json.loads(
        (output_dir / "mmuad_candidate_assignment_block_summary.json").read_text(encoding="utf-8")
    )
    assert len(blocks) >= 5
    assert "duration_s_sum" in summary.columns
    assert payload["block_count"] == len(blocks)
