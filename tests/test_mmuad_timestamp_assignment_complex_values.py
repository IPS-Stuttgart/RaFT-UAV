from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment


@pytest.mark.parametrize(
    ("argument_name", "invalid_values"),
    [
        ("requested_times", [0.0 + 1.0j]),
        ("requested_times", [np.complex128(0.0 + 1.0j)]),
        ("requested_times", [np.array(0.0 + 1.0j)]),
        ("prediction_times", [0.0 + 1.0j]),
        ("prediction_times", [np.complex128(0.0 + 1.0j)]),
        ("prediction_times", [np.array(0.0 + 1.0j)]),
    ],
)
def test_timestamp_assignment_rejects_complex_timestamp_values(
    argument_name: str,
    invalid_values: object,
) -> None:
    arguments = {
        "requested_times": [0.0],
        "prediction_times": [0.0],
        "tolerance_s": 0.0,
    }
    arguments[argument_name] = invalid_values

    with pytest.raises(
        ValueError,
        match=rf"{argument_name} must not contain complex timestamp values",
    ):
        optimal_timestamp_assignment(**arguments)


def test_timestamp_assignment_still_accepts_real_numpy_scalars() -> None:
    assert optimal_timestamp_assignment(
        [np.float32(0.0), np.float64(1.0)],
        np.array([0, 1], dtype=np.int64),
        tolerance_s=0.0,
    ) == {0: 0, 1: 1}
