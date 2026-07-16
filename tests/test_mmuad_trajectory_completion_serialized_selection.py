from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


@pytest.mark.parametrize(
    "false_token",
    ["False", "FALSE", "0", "0.0", "f", "no", "n", "none", "null", "<NA>", ""],
)
def test_trajectory_smoothing_respects_serialized_false_selection_flags(
    false_token: str,
) -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
            "source": ["lidar", "soft", "lidar"],
            "track_id": ["a", "b", "a"],
            "class_name": ["uav", "uav", "uav"],
            "update_action": ["selected_update", "soft_anchor", "selected_update"],
            "selected_path_update": ["true", false_token, "1"],
            "state_x_m": [0.0, 100.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [5.0, 5.0, 5.0],
            "v_x_mps": [0.0, 0.0, 0.0],
            "v_y_mps": [0.0, 0.0, 0.0],
            "v_z_mps": [0.0, 0.0, 0.0],
        }
    )

    result = complete_and_smooth_estimates(
        estimates,
        config=TrajectoryCompletionConfig(
            mode="fixed-lag",
            fixed_lag_s=2.0,
        ),
    )

    middle = result.estimates.loc[result.estimates["time_s"] == 1.0].iloc[0]
    assert float(middle["state_x_m"]) == pytest.approx(1.0)
    assert result.estimates["selected_path_update"].tolist() == [True, False, True]
