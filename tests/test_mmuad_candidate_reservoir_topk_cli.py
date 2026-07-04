from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import main as reservoir_main


def test_explicit_top_k_replaces_candidate_reservoir_defaults(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "reservoir.csv"
    summary_csv = tmp_path / "summary.csv"

    pd.DataFrame(
        [
            {"sequence_id": "seqA", "time_s": 0.0, "source": "lidar", "x_m": 0.0, "y_m": 0.0, "z_m": 1.0, "confidence": 0.1},
            {"sequence_id": "seqA", "time_s": 0.0, "source": "lidar", "x_m": 20.0, "y_m": 0.0, "z_m": 1.0, "confidence": 0.9},
        ]
    ).to_csv(candidates_csv, index=False)
    pd.DataFrame(
        [{"sequence_id": "seqA", "time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0}]
    ).to_csv(truth_csv, index=False)

    status = reservoir_main(
        [
            "--candidate",
            f"candidate={candidates_csv}",
            "--output-csv",
            str(output_csv),
            "--truth-csv",
            str(truth_csv),
            "--oracle-summary-csv",
            str(summary_csv),
            "--top-k",
            "2",
        ]
    )

    assert status == 0
    columns = set(pd.read_csv(summary_csv).columns)
    assert "oracle_top2_3d_m_mse" in columns
    assert "oracle_top1_3d_m_mse" not in columns
    assert "oracle_top3_3d_m_mse" not in columns
    assert "oracle_top5_3d_m_mse" not in columns
    assert "oracle_top10_3d_m_mse" not in columns
    assert "oracle_top20_3d_m_mse" not in columns
