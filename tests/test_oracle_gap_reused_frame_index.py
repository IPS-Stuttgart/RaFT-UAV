from __future__ import annotations

import pandas as pd

from raft_uav.evaluation.oracle_gap_decomposition import (
    OracleGapConfig,
    decompose_radar_oracle_gap,
)


def test_oracle_gap_keeps_reused_frame_indices_physically_separate() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "frame_index": [7, 7],
            "track_id": [101, 202],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    selected = radar.copy()
    selected["association_replay_accepted"] = True
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        selected_radar=selected,
        config=OracleGapConfig(
            plausible_candidate_gate_m=5.0,
            truth_time_gate_s=1.0,
        ),
    )

    assert rows["frame_key_type"].tolist() == [
        "frame_index_time_s",
        "frame_index_time_s",
    ]
    assert rows["frame_key"].nunique() == 2
    assert rows["candidate_count"].tolist() == [1, 1]
    assert rows["nearest_candidate_track_id"].tolist() == [101, 202]
    assert rows["selected_track_id"].tolist() == [101, 202]
    assert rows["selected_error_m"].tolist() == [0.0, 0.0]
    assert rows["category"].tolist() == [
        "correct_candidate_selected",
        "correct_candidate_selected",
    ]


def test_oracle_gap_keeps_candidates_from_one_indexed_frame_grouped() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "frame_index": [7, 7],
            "track_id": [101, 102],
            "east_m": [0.0, 20.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        config=OracleGapConfig(plausible_candidate_gate_m=5.0),
    )

    assert rows["frame_key_type"].tolist() == ["frame_index"]
    assert rows["frame_key"].tolist() == [7]
    assert rows["candidate_count"].tolist() == [2]
