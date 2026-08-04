from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


def _estimates(times: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1"] * len(times),
            "time_s": times,
            "selected_path_update": [True] * len(times),
            "state_x_m": times,
            "state_y_m": [0.0] * len(times),
            "state_z_m": [5.0] * len(times),
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [5.0, 5.0, 5.0],
        }
    )


@pytest.mark.parametrize(
    "serialized_false",
    ["False", "0", "off", np.asarray("False")],
)
def test_serialized_false_disables_truth_timestamp_insertion(
    serialized_false: object,
) -> None:
    result = complete_and_smooth_estimates(
        _estimates([0.0, 2.0]),
        _truth(),
        config=TrajectoryCompletionConfig(
            mode="gap-interpolation",
            max_gap_s=3.0,
            include_truth_timestamps=serialized_false,
            infer_missing_grid=False,
        ),
    )

    assert result.estimates["time_s"].tolist() == [0.0, 2.0]


@pytest.mark.parametrize("serialized_false", ["False", "0", "off"])
def test_serialized_false_disables_missing_grid_inference(
    serialized_false: object,
) -> None:
    result = complete_and_smooth_estimates(
        _estimates([0.0, 1.0, 2.0, 4.0]),
        config=TrajectoryCompletionConfig(
            mode="gap-interpolation",
            max_gap_s=3.0,
            include_truth_timestamps=False,
            infer_missing_grid=serialized_false,
        ),
    )

    assert result.estimates["time_s"].tolist() == [0.0, 1.0, 2.0, 4.0]


def test_invalid_serialized_boolean_control_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="include_truth_timestamps must be a Boolean scalar",
    ):
        complete_and_smooth_estimates(
            _estimates([0.0, 2.0]),
            config=TrajectoryCompletionConfig(
                mode="gap-interpolation",
                include_truth_timestamps="sometimes",
            ),
        )
