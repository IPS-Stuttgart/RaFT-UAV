from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


@pytest.mark.parametrize("false_value", ["False", "0", "off", "no"])
def test_serialized_false_selected_updates_do_not_influence_smoothing(
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
    assert float(middle["state_x_m"]) == pytest.approx(1.0)
    assert result.estimates["selected_path_update"].tolist() == [True, False, True]


def test_selected_update_parser_accepts_zero_dimensional_boolean_array() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
            "selected_path_update": [True, np.array(False), True],
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
    assert float(middle["state_x_m"]) == pytest.approx(1.0)
