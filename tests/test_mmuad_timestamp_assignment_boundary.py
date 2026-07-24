from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment


@pytest.mark.parametrize(
    ("request_time", "prediction_time", "boundary_side"),
    [
        (np.float64(1.99999999999999), np.float64(-1.0e-14), "lower"),
        (np.float64(-1.99999999999999), np.float64(1.0e-14), "upper"),
    ],
)
def test_timestamp_assignment_keeps_rounded_boundary_matches(
    request_time: np.float64,
    prediction_time: np.float64,
    boundary_side: str,
) -> None:
    tolerance_s = 2.0
    if boundary_side == "lower":
        assert prediction_time < request_time - tolerance_s
    else:
        assert prediction_time > request_time + tolerance_s
    assert abs(float(prediction_time - request_time)) == tolerance_s

    assignment = optimal_timestamp_assignment(
        [request_time],
        [prediction_time],
        tolerance_s=tolerance_s,
    )

    assert assignment == {0: 0}
