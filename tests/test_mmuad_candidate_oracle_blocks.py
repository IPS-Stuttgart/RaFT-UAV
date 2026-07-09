from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_oracle_blocks import (
    build_candidate_oracle_block_tables,
    main as oracle_blocks_main,
)


def _frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6 + ["seqB"] * 2,
            "time_s": [0.0, 0.5, 1.0, 5.0, 5.5, 6.0, 0.0, 0.5],
            "candidate_count": [10, 10, 10, 10, 10, 10, 5, 5],
            "oracle_all_3d_m": [0.5, 0.4, 0.6, 9.0, 10.0, 0.7, 0.5, 0.6],
            "oracle_all_rank": [1, 8, 9, 2, 3, 1, 2, 2],
            "oracle_in_top3": [True, False, False, True, True, True, True, True],
            "oracle_all_candidate_branch": ["raw", "raw", "raw", "dynamic", "dynamic", "raw", "raw", "raw"],
            "oracle_all_candidate_source": ["lidar", "lidar", "lidar", "livox", "livox", "lidar", "lidar", "lidar"],
        }
    )


def test_candidate_oracle_blocks_split_missing_and_buried_modes() -> None:
    blocks, summary = build_candidate_oracle_block_tables(
        _frame_rows(),
        oracle_error_threshold_m=5.0,
        top_k=3,
        max_gap_s=1.0,
    )

    assert set(blocks["oracle_failure_mode"]) == {
        "covered_in_topk",
        "good_candidate_buried",
        "missing_good_candidate",
    }
    buried = blocks.loc[blocks["oracle_failure_mode"] == "good_candidate_buried"].iloc[0]
    assert buried["frame_count"] == 2
    assert buried["dominant_oracle_branch"] == "raw"
    missing = blocks.loc[blocks["oracle_failure_mode"] == "missing_good_candidate"].iloc[0]
    assert missing["frame_count"] == 2
    assert missing["oracle_all_3d_m_max"] == 10.0
    pooled_modes = set(summary.loc[summary["sequence_id"] == "__pooled__", "oracle_failure_mode"])
    assert "missing_good_candidate" in pooled_modes
    assert "good_candidate_buried" in pooled_modes


def test_candidate_oracle_blocks_preserve_numeric_true_topk_flags() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqC", "seqC", "seqC", "seqC"],
            "time_s": [0.0, 0.5, 1.0, 1.5],
            "oracle_all_3d_m": [0.2, 0.3, 0.4, 0.5],
            "oracle_all_rank": [1, 2, 4, 5],
            "oracle_in_top3": [1.0, "1.0", 0.0, "0.0"],
        }
    )

    blocks, _ = build_candidate_oracle_block_tables(
        rows,
        oracle_error_threshold_m=5.0,
        top_k=3,
        max_gap_s=1.0,
    )

    covered = blocks.loc[blocks["oracle_failure_mode"] == "covered_in_topk"].iloc[0]
    buried = blocks.loc[blocks["oracle_failure_mode"] == "good_candidate_buried"].iloc[0]
    assert int(covered["frame_count"]) == 2
    assert int(buried["frame_count"]) == 2


def test_candidate_oracle_blocks_cli_writes_artifacts(tmp_path) -> None:
    frame_csv = tmp_path / "frames.csv"
    output_dir = tmp_path / "out"
    _frame_rows().to_csv(frame_csv, index=False)

    rc = oracle_blocks_main(
        [
            "--frame-csv",
            str(frame_csv),
            "--output-dir",
            str(output_dir),
            "--oracle-error-threshold-m",
            "5",
            "--top-k",
            "3",
            "--max-gap-s",
            "1",
        ]
    )

    assert rc == 0
    blocks = pd.read_csv(output_dir / "mmuad_candidate_oracle_blocks.csv")
    summary = pd.read_csv(output_dir / "mmuad_candidate_oracle_block_summary.csv")
    assert len(blocks) >= 4
    assert "duration_s_sum" in summary.columns
    assert (output_dir / "mmuad_candidate_oracle_block_summary.json").exists()
