from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad import track5_jerk_limit
from raft_uav.mmuad._repair_complex_row_validation_patch import _is_complex_scalar


def _boxed_scalar(value: object) -> np.ndarray:
    boxed = np.empty((), dtype=object)
    boxed[()] = value
    return boxed


def test_complex_guard_unwraps_recursive_object_scalars() -> None:
    value = _boxed_scalar(_boxed_scalar(np.complex128(2.0 + 3.0j)))

    assert _is_complex_scalar(value)


def test_jerk_control_rejects_recursively_wrapped_numpy_complex() -> None:
    value = _boxed_scalar(_boxed_scalar(np.complex128(2.0 + 3.0j)))

    with pytest.raises(ValueError, match="max_jerk_mps3"):
        track5_jerk_limit._finite_scalar(
            value,
            message="max_jerk_mps3 must be positive and finite",
        )


def test_jerk_control_preserves_recursively_wrapped_real_scalars() -> None:
    value = _boxed_scalar(_boxed_scalar(np.float64(2.5)))

    assert track5_jerk_limit._finite_scalar(value, message="invalid") == 2.5
