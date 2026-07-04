from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_pool_compare import main as pool_compare_main


def test_candidate_pool_compare_cli_explicit_top_k_replaces_defaults(tmp_path) -> None:
    reference_csv = tmp_path / "reference.csv"
    pruned_csv = tmp_path / "pruned.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["lidar", "livox", "lidar", "livox"],
            "track_id": ["good-0", "bad-0", "good-1", "bad-1"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "x_m": [0.0, 20.0, 1.0, 18.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "candidate_reservoir_score": [0.1, 0.9, 0.2, 0.95],
            "ranker_score": [0.1, 0.9, 0.2, 0.95],
        }
    ).to_csv(reference_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "source": ["livox", "livox"],
            "track_id": ["bad-0", "bad-1"],
            "candidate_branch": ["translated", "translated"],
            "x_m": [20.0, 18.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "candidate_reservoir_score": [0.9, 0.95],
            "ranker_score": [0.9, 0.95],
        }
    ).to_csv(pruned_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    ).to_csv(truth_csv, index=False)

    assert (
        pool_compare_main(
            [
                "--reference-candidate",
                f"full={reference_csv}",
                "--candidate",
                f"pruned={pruned_csv}",
                "--truth-csv",
                str(truth_csv),
                "--output-dir",
                str(output_dir),
                "--top-k",
                "1",
                "--top-k",
                "2",
                "--max-truth-time-delta-s",
                "0.1",
            ]
        )
        == 0
    )

    pooled = pd.read_csv(output_dir / "mmuad_candidate_pool_compare_pooled.csv")
    frame_rows = pd.read_csv(output_dir / "mmuad_candidate_pool_compare_frames.csv")
    assert "reference_oracle_top2_3d_m" in frame_rows.columns
    assert "reference_oracle_top3_3d_m" not in frame_rows.columns
    assert "reference_oracle_top3_mse" not in pooled.columns
