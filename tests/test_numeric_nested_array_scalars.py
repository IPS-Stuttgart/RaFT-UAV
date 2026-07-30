from __future__ import annotations

import warnings

import numpy as np
import pytest

from raft_uav.numeric import optional_float, optional_int


def _object_scalar(payload: object) -> np.ndarray:
    value = np.empty((), dtype=object)
    value[()] = payload
    return value


@pytest.mark.parametrize(
    "payload",
    [
        np.array([1.25]),
        np.array([[1.25]]),
    ],
)
def test_optional_numeric_rejects_nested_non_scalar_arrays_without_coercion(
    payload: np.ndarray,
) -> None:
    value = _object_scalar(payload)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert optional_float(value) is None
        assert optional_int(value) is None


def test_optional_float_accepts_nested_zero_dimensional_real_array() -> None:
    value = _object_scalar(np.array(1.25))

    assert optional_float(value) == 1.25


def test_optional_numeric_rejects_self_referential_object_scalar() -> None:
    value = np.empty((), dtype=object)
    value[()] = value

    assert optional_float(value) is None
    assert optional_int(value) is None
