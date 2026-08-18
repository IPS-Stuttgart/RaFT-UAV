from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


def test_trajectory_completion_does_not_merge_reused_tracks_across_flights() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared", "shared", "shared"],
            "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "output_track_id": ["track-1", "track-1", "track-1", "track-1"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "update_action": [
                "selected_update",
                "selected_update",
                "selected_update",
                "selected_update",
            ],
            "selected_path_update": [True, True, True, True],
            "state_x_m": [0.0, 1.0, 100.0, 101.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [5.0, 5.0, 5.0, 5.0],
            "v_x_mps": [1.0, 1.0, 1.0, 1.0],
            "v_y_mps": [0.0, 0.0, 0.0, 0.0],
            "v_z_mps": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = complete_and_smooth_estimates(
        estimates,
        config=TrajectoryCompletionConfig(
            mode="none",
            include_truth_timestamps=False,
            infer_missing_grid=False,
        ),
    )

    completed = result.estimates.sort_values(["flight_id", "time_s"]).reset_index(
        drop=True
    )
    assert len(completed) == 4
    assert completed["flight_id"].tolist() == [
        "flight-a",
        "flight-a",
        "flight-b",
        "flight-b",
    ]
    assert completed["state_x_m"].tolist() == [0.0, 1.0, 100.0, 101.0]
    assert len(result.speed_gate_summary) == 2
