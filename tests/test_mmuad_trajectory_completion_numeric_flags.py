from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    _parse_selected_path_update,
    complete_and_smooth_estimates,
)


@pytest.mark.parametrize(
    "value",
    [2, -1, 0.5, np.int64(2), np.float64(-0.5)],
)
def test_selected_path_update_rejects_non_binary_numeric_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="selected_path_update"):
        _parse_selected_path_update(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, False),
        (1, True),
        (0.0, False),
        (1.0, True),
        (np.inf, False),
        (-np.inf, False),
    ],
)
def test_selected_path_update_preserves_supported_numeric_values(
    value: object,
    expected: bool,
) -> None:
    assert _parse_selected_path_update(value) is expected


def test_non_binary_selection_flag_cannot_enter_smoothing_fit() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
            "selected_path_update": [1, 2, 1],
            "state_x_m": [0.0, 10.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [5.0, 5.0, 5.0],
        }
    )

    with pytest.raises(ValueError, match="selected_path_update"):
        complete_and_smooth_estimates(
            estimates,
            config=TrajectoryCompletionConfig(
                mode="fixed-lag",
                fixed_lag_s=2.0,
            ),
        )
