from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment


@pytest.mark.parametrize(
    ("requested_times", "prediction_times"),
    [
        ([True], [1.0]),
        ([1.0], [False]),
        ([np.bool_(True)], [1.0]),
        ([1.0], [np.bool_(False)]),
    ],
)
def test_optimal_timestamp_assignment_rejects_boolean_timestamps(
    requested_times: list[object],
    prediction_times: list[object],
) -> None:
    with pytest.raises(
        ValueError,
        match="timestamp arrays must contain only finite values",
    ):
        optimal_timestamp_assignment(
            requested_times,
            prediction_times,
            tolerance_s=0.0,
        )


def test_optimal_timestamp_assignment_keeps_numeric_zero_one_timestamps() -> None:
    assert optimal_timestamp_assignment(
        [0.0, 1.0],
        [0, 1],
        tolerance_s=0.0,
    ) == {0: 0, 1: 1}
