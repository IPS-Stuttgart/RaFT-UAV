from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment


def _nested_zero_dimensional(value: object, *, depth: int = 2) -> np.ndarray:
    nested = value
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = nested
        nested = wrapper
    assert isinstance(nested, np.ndarray)
    return nested


@pytest.mark.parametrize("invalid_side", ["requested", "prediction"])
def test_assignment_rejects_nested_one_element_timestamp_vectors(
    invalid_side: str,
) -> None:
    invalid = _nested_zero_dimensional(np.array([0.0]))
    requested = [invalid] if invalid_side == "requested" else [0.0]
    predictions = [invalid] if invalid_side == "prediction" else [0.0]

    with pytest.raises(ValueError, match="must be one-dimensional"):
        optimal_timestamp_assignment(
            requested,
            predictions,
            tolerance_s=0.0,
        )


def test_assignment_rejects_nested_one_element_tolerance_vector() -> None:
    invalid = _nested_zero_dimensional(np.array([0.0]))

    with pytest.raises(
        ValueError,
        match="tolerance_s must be a finite nonnegative real scalar",
    ):
        optimal_timestamp_assignment(
            [0.0],
            [0.0],
            tolerance_s=invalid,
        )


def test_assignment_accepts_recursively_nested_scalar_arrays() -> None:
    assignment = optimal_timestamp_assignment(
        [_nested_zero_dimensional(0.0, depth=3)],
        [_nested_zero_dimensional(0.0, depth=3)],
        tolerance_s=_nested_zero_dimensional(0.0, depth=3),
    )

    assert assignment == {0: 0}


def test_assignment_rejects_cyclic_scalar_array_payloads() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(
        ValueError,
        match="requested_times must contain only finite real scalar timestamp values",
    ):
        optimal_timestamp_assignment(
            [cyclic],
            [0.0],
            tolerance_s=0.0,
        )
