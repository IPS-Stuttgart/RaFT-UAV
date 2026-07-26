from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment


@pytest.mark.parametrize(
    "invalid_timestamp",
    [
        True,
        np.bool_(False),
        np.array(True),
        1.0 + 0.0j,
        np.complex64(1.0 + 0.0j),
        np.array(1.0 + 0.0j),
        np.ma.masked,
    ],
)
@pytest.mark.parametrize("invalid_side", ["requested", "prediction"])
def test_assignment_rejects_invalid_timestamp_values(
    invalid_timestamp: object,
    invalid_side: str,
) -> None:
    requested_times = [invalid_timestamp] if invalid_side == "requested" else [0.0]
    prediction_times = [invalid_timestamp] if invalid_side == "prediction" else [0.0]

    with pytest.raises(
        ValueError,
        match="timestamp arrays must contain only finite real scalar values",
    ):
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
