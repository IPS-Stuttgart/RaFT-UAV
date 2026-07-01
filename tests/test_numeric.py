from __future__ import annotations

import numpy as np
import pytest

from raft_uav.numeric import optional_float, optional_int


class _FloatableArrayLike:
    ndim = 1

    def __float__(self) -> float:
        return 1.25


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not-a-number",
        "nan",
        "inf",
        "-inf",
        float("nan"),
        float("inf"),
        object(),
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.array(True),
        np.array(False),
        1 + 0j,
        np.complex64(1 + 0j),
        np.complex128(1 + 2j),
        np.array(1 + 0j),
    ],
)
def test_optional_float_rejects_absent_malformed_nonfinite_boolean_and_complex_values(
    value: object,
) -> None:
    assert optional_float(value) is None


@pytest.mark.parametrize("value", [np.array([1.0]), _FloatableArrayLike()])
def test_optional_float_rejects_non_scalar_array_like_values(value: object) -> None:
    assert optional_float(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (1.25, 1.25),
        ("3.5", 3.5),
        (np.array(2.5), 2.5),
    ],
)
def test_optional_float_accepts_finite_scalar_values(value: object, expected: float) -> None:
    assert optional_float(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "nan",
        float("nan"),
        float("inf"),
        object(),
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.array(True),
        np.array(False),
        "3.5",
        4.9,
        -2.1,
        4 + 0j,
        np.complex128(4 + 0j),
        np.array(4 + 0j),
    ],
)
def test_optional_int_rejects_absent_malformed_nonfinite_boolean_fractional_and_complex_values(
    value: object,
) -> None:
    assert optional_int(value) is None


@pytest.mark.parametrize("value", [np.array([1]), _FloatableArrayLike()])
def test_optional_int_rejects_non_scalar_array_like_values(value: object) -> None:
    assert optional_int(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("4", 4), (4, 4), (4.0, 4), (-2.0, -2), (np.int64(7), 7), (np.array(8), 8)],
)
def test_optional_int_accepts_integer_like_finite_scalar_values(value: object, expected: int) -> None:
    assert optional_int(value) == expected
