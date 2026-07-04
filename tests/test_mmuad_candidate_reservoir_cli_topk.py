from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import main as reservoir_main


def test_candidate_reservoir_cli_explicit_top_k_replaces_defaults(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "reservoir.csv"
    summary_csv = tmp_path / "oracle_summary.csv"

    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["lidar_360", "lidar_360", "lidar_360", "livox_avia"],
            "track_id": ["raw-good", "calib-bad", "raw-good-1", "calib-bad-1"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "x_m": [0.0, 20.0, 1.0, 18.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "confidence": [0.1, 0.99, 0.2, 0.95],
        }
    ).to_csv(candidates_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    ).to_csv(truth_csv, index=False)

    status = reservoir_main(
        [
            "--candidate",
            f"raw={candidates_csv}",
            "--output-csv",
            str(output_csv),
            "--truth-csv",
            str(truth_csv),
            "--oracle-summary-csv",
            str(summary_csv),
            "--top-k",
            "2",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    summary = pd.read_csv(summary_csv)
    assert "oracle_top2_3d_m_mse" in summary.columns
    assert "oracle_top1_3d_m_mse" not in summary.columns
    assert "oracle_top3_3d_m_mse" not in summary.columns
    assert "oracle_top5_3d_m_mse" not in summary.columns
    assert "oracle_top10_3d_m_mse" not in summary.columns
    assert "oracle_top20_3d_m_mse" not in summary.columns
