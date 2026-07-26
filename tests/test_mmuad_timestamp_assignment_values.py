from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment


@pytest.mark.parametrize(
    ("invalid_timestamp", "expected_message"),
    [
        (True, "must not contain Boolean timestamp values"),
        (np.bool_(False), "must not contain Boolean timestamp values"),
        (np.array(True), "must not contain Boolean timestamp values"),
        (1.0 + 0.0j, "must contain only finite real scalar timestamp values"),
        (np.complex64(1.0 + 0.0j), "must contain only finite real scalar timestamp values"),
        (np.array(1.0 + 0.0j), "must contain only finite real scalar timestamp values"),
        (np.ma.masked, "must contain only finite real scalar timestamp values"),
    ],
)
@pytest.mark.parametrize("invalid_side", ["requested", "prediction"])
def test_assignment_rejects_invalid_timestamp_values(
    invalid_timestamp: object,
    expected_message: str,
    invalid_side: str,
) -> None:
    requested_times = [invalid_timestamp] if invalid_side == "requested" else [0.0]
    prediction_times = [invalid_timestamp] if invalid_side == "prediction" else [0.0]

    with pytest.raises(ValueError, match=expected_message):
        optimal_timestamp_assignment(
            requested_times,
            prediction_times,
            tolerance_s=0.0,
        )


def test_assignment_accepts_zero_dimensional_real_timestamp_arrays() -> None:
    assignment = optimal_timestamp_assignment(
        [np.array(0.0)],
        [np.array(0.0)],
        tolerance_s=0.0,
    )

    assert assignment == {0: 0}
