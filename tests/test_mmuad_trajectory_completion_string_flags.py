from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


@pytest.mark.parametrize("false_value", ["False", "0", "off", "no"])
def test_string_false_selected_path_updates_do_not_influence_smoothing(
    false_value: str,
) -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
            "selected_path_update": ["True", false_value, "True"],
            "state_x_m": [0.0, 10.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [5.0, 5.0, 5.0],
        }
    )

    result = complete_and_smooth_estimates(
        estimates,
        config=TrajectoryCompletionConfig(mode="fixed-lag", fixed_lag_s=2.0),
    )

    middle = result.estimates.loc[result.estimates["time_s"] == 1.0].iloc[0]
    assert abs(float(middle["state_x_m"]) - 1.0) < 1.0e-6
    assert result.estimates["selected_path_update"].tolist() == [True, False, True]
