from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_grid import (
    main as reservoir_grid_main,
    run_candidate_reservoir_offset_grid,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["lidar_360", "livox_avia", "lidar_360", "livox_avia"],
            "track_id": ["raw-good-0", "translated-bad-0", "raw-good-1", "translated-bad-1"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "x_m": [0.0, 20.0, 1.0, 18.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "ranker_score": [0.10, 0.90, 0.20, 0.95],
            "confidence": [0.10, 0.90, 0.20, 0.95],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    )


def test_score_offset_grid_can_promote_low_score_raw_branch() -> None:
    summary, best = run_candidate_reservoir_offset_grid(
        _candidate_rows(),
        truth=_truth_rows(),
        branch_offset_grid=["raw=0,1"],
        global_top_n=1,
        per_source_top_n=0,
        per_branch_top_n=0,
        max_candidates_per_frame=1,
        top_k_values=(1,),
        max_truth_time_delta_s=0.1,
        selection_metric="oracle_top1_3d_m_mse",
        write_best_reservoir=True,
    )

    assert summary.iloc[0]["grid_label"] == "branch_raw_1"
    assert summary.iloc[0]["oracle_top1_3d_m_mse"] == 0.0
    assert best is not None
    assert set(best["track_id"]) == {"raw-good-0", "raw-good-1"}
    assert "candidate_reservoir_grid_branch_offset" in best.columns


def test_candidate_reservoir_offset_grid_cli_writes_summary_and_best(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _candidate_rows().to_csv(candidate_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    rc = reservoir_grid_main(
        [
            "--candidate",
            f"mixed={candidate_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--branch-score-offset-grid",
            "raw=0,1",
            "--global-top-n",
            "1",
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "0",
            "--max-candidates-per-frame",
            "1",
            "--top-k",
            "1",
            "--max-truth-time-delta-s",
            "0.1",
            "--selection-metric",
            "oracle_top1_3d_m_mse",
            "--write-best-reservoir",
        ]
    )

    assert rc == 0
    summary = pd.read_csv(output_dir / "mmuad_candidate_reservoir_offset_grid_summary.csv")
    best = pd.read_csv(output_dir / "best_candidate_reservoir.csv")
    assert summary.iloc[0]["grid_label"] == "branch_raw_1"
    assert summary.iloc[0]["oracle_top1_3d_m_mse"] == 0.0
    assert set(best["track_id"]) == {"raw-good-0", "raw-good-1"}
