from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


def test_trajectory_completion_keeps_final_same_timestamp_posterior() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["sequence"] * 4,
            "time_s": [0.0, 1.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 9.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
            "selected_path_update": [True, False, True, True],
            "update_action": [
                "selected_update",
                "soft_anchor",
                "selected_update",
                "selected_update",
            ],
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

    at_duplicate_time = result.estimates.loc[result.estimates["time_s"] == 1.0]
    assert len(at_duplicate_time) == 1
    assert at_duplicate_time.iloc[0]["state_x_m"] == 9.0
    assert at_duplicate_time.iloc[0]["update_action"] == "selected_update"
