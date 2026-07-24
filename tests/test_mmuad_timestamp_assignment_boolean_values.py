from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment


@pytest.mark.parametrize(
    ("argument_name", "invalid_value"),
    [
        ("requested_times", True),
        ("requested_times", np.bool_(False)),
        ("prediction_times", True),
        ("prediction_times", np.bool_(False)),
    ],
)
def test_timestamp_assignment_rejects_boolean_timestamp_values(
    argument_name: str,
    invalid_value: object,
) -> None:
    arguments = {
        "requested_times": [0.0],
        "prediction_times": [0.0],
        "tolerance_s": 0.0,
    }
    arguments[argument_name] = [invalid_value]

    with pytest.raises(
        ValueError,
        match=rf"{argument_name} must not contain Boolean timestamp values",
    ):
        optimal_timestamp_assignment(**arguments)


def test_timestamp_assignment_still_accepts_numeric_zero_and_one() -> None:
    assert optimal_timestamp_assignment(
        [0, 1],
        [0.0, 1.0],
        tolerance_s=0.0,
    ) == {0: 0, 1: 1}
