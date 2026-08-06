from __future__ import annotations

import pytest

from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment


@pytest.mark.parametrize("invalid_sequence", ["10", b"10", bytearray(b"10")])
@pytest.mark.parametrize("invalid_side", ["requested", "prediction"])
def test_assignment_rejects_scalar_text_or_byte_sequences(
    invalid_sequence: object,
    invalid_side: str,
) -> None:
    requested_times = invalid_sequence if invalid_side == "requested" else [1.0, 0.0]
    prediction_times = invalid_sequence if invalid_side == "prediction" else [1.0, 0.0]

    with pytest.raises(
        ValueError,
        match=rf"{invalid_side}_times must be one-dimensional",
    ):
        optimal_timestamp_assignment(
            requested_times,
            prediction_times,
            tolerance_s=0.0,
        )


def test_assignment_preserves_iterables_of_numeric_text_values() -> None:
    assignment = optimal_timestamp_assignment(
        ["1", "0"],
        ["1.0", "0.0"],
        tolerance_s=0.0,
    )

    assert assignment == {0: 0, 1: 1}
