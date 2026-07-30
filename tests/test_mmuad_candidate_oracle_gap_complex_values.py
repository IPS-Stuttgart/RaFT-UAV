from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_oracle_gap import build_candidate_oracle_gap


def test_oracle_gap_rejects_nonreal_candidates_without_losing_real_rows() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 0.0],
            "source": ["lidar_360", "lidar_360"],
            "track_id": ["nonreal", "real"],
            "x_m": [0.1 + 2.0j, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [0.9, 0.8],
        }
    )
    selected = candidates.iloc[[1]].copy()

    rows = build_candidate_oracle_gap(
        candidates,
        selected,
        truth,
        max_time_delta_s=0.1,
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["nearest_candidate_track_id"] == "real"
    assert int(row["candidate_count_at_nearest_time"]) == 1
    assert float(row["nearest_minus_truth_error_m"]) == 2.0
    assert bool(row["selected_candidate_found"])


def test_oracle_gap_rejects_nonreal_truth_without_losing_real_rows() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0 + 3.0j],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "source": ["lidar_360", "lidar_360"],
            "track_id": ["t0", "t1"],
            "x_m": [2.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [0.8, 0.8],
        }
    )

    rows = build_candidate_oracle_gap(
        candidates,
        candidates,
        truth,
        max_time_delta_s=0.1,
    )

    assert rows["time_s"].tolist() == [0.0]
