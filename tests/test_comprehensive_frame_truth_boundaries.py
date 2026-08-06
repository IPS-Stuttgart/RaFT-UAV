from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.comprehensive_improvements import candidate_recall_regret_table


def test_reused_frame_counter_keeps_physical_frames_and_selected_rows_separate() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "frame_index": [7, 7],
            "track_id": [11, 22],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    table = candidate_recall_regret_table(
        radar,
        truth,
        selected_radar=radar,
        truth_gate_m=1.0,
        truth_time_gate_s=0.1,
    )

    assert table["time_s"].tolist() == [0.0, 10.0]
    assert table["frame_key_type"].tolist() == ["frame_index", "frame_index"]
    assert table["frame_key"].tolist() == [7, 7]
    assert table["candidate_rows"].tolist() == [1, 1]
    assert table["candidate_available"].astype(bool).tolist() == [True, True]
    assert table["selected_track_id"].astype(int).tolist() == [11, 22]
    assert table["selected_error_m"].tolist() == [0.0, 0.0]


def test_missing_frame_timestamp_does_not_match_an_arbitrary_truth_row() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [np.nan],
            "frame_index": [np.nan],
            "track_id": [9],
            "east_m": [100.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    table = candidate_recall_regret_table(
        radar,
        truth,
        truth_gate_m=1.0,
        truth_time_gate_s=0.1,
    )

    assert len(table) == 1
    assert not bool(table.loc[0, "truth_available"])
    assert table.loc[0, "failure_bucket"] == "no_nearby_truth"


def test_invalid_truth_position_is_skipped_for_nearest_finite_sample() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0],
            "frame_index": [1],
            "track_id": [5],
            "east_m": [1.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "east_m": [np.nan, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    table = candidate_recall_regret_table(
        radar,
        truth,
        truth_gate_m=1.0,
        truth_time_gate_s=0.2,
    )

    assert bool(table.loc[0, "truth_available"])
    assert np.isclose(table.loc[0, "truth_time_delta_s"], 0.1)
    assert bool(table.loc[0, "candidate_available"])
    assert np.isclose(table.loc[0, "best_candidate_error_m"], 0.0)
