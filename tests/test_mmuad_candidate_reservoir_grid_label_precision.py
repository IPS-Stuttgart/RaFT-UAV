from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_grid import run_candidate_reservoir_offset_grid


def test_high_precision_offset_labels_select_the_matching_best_reservoir() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["lidar_360", "livox_avia"],
            "track_id": ["raw-good", "translated-bad"],
            "candidate_branch": ["raw", "translated"],
            "x_m": [0.0, 20.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "ranker_score": [0.1, 0.90000015],
            "confidence": [0.1, 0.90000015],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )

    summary, best = run_candidate_reservoir_offset_grid(
        candidates,
        truth=truth,
        branch_offset_grid=["raw=0.8000002,0.8000001"],
        global_top_n=1,
        per_source_top_n=0,
        per_branch_top_n=0,
        max_candidates_per_frame=1,
        top_k_values=(1,),
        max_truth_time_delta_s=0.1,
        selection_metric="oracle_top1_3d_m_mse",
        write_best_reservoir=True,
    )

    assert summary["grid_label"].is_unique
    assert summary.iloc[0]["grid_label"] == "branch_raw_0p8000002"
    assert best is not None
    assert best["track_id"].tolist() == ["raw-good"]
