from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from raft_uav.evaluation.metrics import position_errors_m


def _object_scalar(value: object) -> object:
    return np.asarray(value, dtype=object)


def _unmasked_object_scalar(value: object) -> object:
    return np.ma.array(np.asarray(value, dtype=object), mask=False)


@pytest.mark.parametrize("complex_type", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "wrap",
    [_object_scalar, _unmasked_object_scalar],
    ids=["object-array", "unmasked-object-array"],
)
def test_position_errors_rejects_object_wrapped_complex_time_gate(
    complex_type: type[np.complexfloating],
    wrap: Callable[[object], object],
) -> None:
    times = np.array([0.0, 1.0])
    positions = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times)]
    )
    max_time_delta_s = wrap(complex_type(complex(0.25, 1.0)))

    with pytest.raises(
        ValueError,
        match="max_time_delta_s must be a finite, non-negative scalar",
    ):
        position_errors_m(
  times,
  positions,
  times,
  positions,
  max_time_delta_s=max_time_delta_s,
        )


def test_position_errors_accepts_object_wrapped_real_time_gate() -> None:
    times = np.array([0.0, 1.0])
    positions = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times)]
    )

    errors = position_errors_m(
        times,
        positions,
        times,
        positions,
        max_time_delta_s=np.asarray(np.float64(0.25), dtype=object),
    )

    np.testing.assert_allclose(errors, np.zeros(2))
