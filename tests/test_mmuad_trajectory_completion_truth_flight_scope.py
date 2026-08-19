from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


def test_completion_truth_errors_do_not_cross_reused_sequence_ids_between_flights() -> None:
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
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared", "shared", "shared"],
            "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "x_m": [0.0, 1.0, 100.0, 101.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [5.0, 5.0, 5.0, 5.0],
        }
    )

    result = complete_and_smooth_estimates(
        estimates,
        truth,
        config=TrajectoryCompletionConfig(
            mode="none",
            include_truth_timestamps=False,
            infer_missing_grid=False,
        ),
    )

    completed = result.estimates.sort_values(["flight_id", "time_s"]).reset_index(
        drop=True
    )
    np.testing.assert_allclose(completed["error_3d_m"].to_numpy(float), 0.0)
    np.testing.assert_allclose(completed["truth_x_m"].to_numpy(float), [0.0, 1.0, 100.0, 101.0])

    ablation = result.smoothing_ablation.loc[
        result.smoothing_ablation["sequence_id"] == "shared"
    ]
    np.testing.assert_allclose(ablation["mean_3d_m"].to_numpy(float), 0.0)

    summary = result.sequence_error_summary.loc[
        result.sequence_error_summary["sequence_id"] == "shared"
    ].iloc[0]
    assert float(summary["raw_mean_3d_m"]) == 0.0
    assert float(summary["final_mean_3d_m"]) == 0.0
