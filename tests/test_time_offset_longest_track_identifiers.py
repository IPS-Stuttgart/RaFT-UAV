from __future__ import annotations

import pandas as pd

import raft_uav.diagnostics.time_offset as time_offset


def test_longest_track_id_ignores_fractional_and_boolean_identifiers() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [12.75, 12.75, True, True, "9", 9.0, "9.0"],
        }
    )

    assert time_offset._longest_track_id(radar) == 9


def test_longest_track_sweep_does_not_select_a_truncated_identifier() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, 0, 0, 1, 1, 1],
            "time_s": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "track_id": [12.75, 12.75, 9, 12.75, 12.75, 9],
            "cat_prob_uav": [0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
            "east_m": [100.0, 200.0, 0.0, 100.0, 200.0, 1.0],
            "north_m": [100.0, 200.0, 0.0, 100.0, 200.0, 0.0],
            "up_m": [0.0] * 6,
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    sweep = time_offset.sweep_radar_against_truth(
        radar=radar,
        truth=truth,
        taus_s=[0.0],
        dimensions=2,
        selection="longest-track",
        catprob_threshold=0.4,
        max_truth_time_delta_s=0.1,
    )

    row = sweep.iloc[0]
    assert row["selected_count"] == 2
    assert row["matched_count"] == 2
    assert row["rmse_error_m"] == 0.0
