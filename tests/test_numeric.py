from __future__ import annotations

import numpy as np
import pytest

from raft_uav.numeric import optional_float, optional_int


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
    ],
)
def test_optional_float_rejects_absent_malformed_nonfinite_and_boolean_values(
    value: object,
) -> None:
    assert optional_float(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (1.25, 1.25),
        ("3.5", 3.5),
    ],
)
def test_optional_float_accepts_finite_values(value: object, expected: float) -> None:
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
        4.9,
        -2.1,
    ],
)
def test_optional_int_rejects_absent_malformed_nonfinite_boolean_and_fractional_values(
    value: object,
) -> None:
    assert optional_int(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("4", 4), (4, 4), (4.0, 4), (-2.0, -2), (np.int64(7), 7)],
)
def test_optional_int_accepts_integer_like_finite_values(value: object, expected: int) -> None:
    assert optional_int(value) == expected
