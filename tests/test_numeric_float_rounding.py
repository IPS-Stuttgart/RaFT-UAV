from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from raft_uav.numeric import optional_int


def test_optional_int_rejects_fractional_text_that_float_would_round() -> None:
    fractional = f"{2 ** 53}.5"

    assert optional_int(fractional) is None
    assert optional_int(Decimal(fractional)) is None


def test_optional_int_preserves_integer_text_that_exceeds_float_precision() -> None:
    integer_text = str((2 ** 53) + 1)
    expected = int(integer_text)

    assert optional_int(integer_text) == expected
    assert optional_int(Decimal(integer_text)) == expected


def test_optional_int_rejects_exact_fraction_that_float_would_round() -> None:
    fractional = Fraction((2**53) + 1, 2)

    assert optional_int(fractional) is None


def test_optional_int_preserves_integral_fraction_beyond_float_precision() -> None:
    expected = (2**53) + 1

    assert optional_int(Fraction(expected, 1)) == expected


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
    reason="np.longdouble does not provide extra precision on this platform",
)
def test_optional_int_preserves_extended_precision_numpy_integer() -> None:
    expected = (2**53) + 1
    value = np.longdouble(str(expected))

    assert optional_int(value) == expected
    assert optional_int(np.array(value)) == expected


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
    reason="np.longdouble does not provide extra precision on this platform",
)
def test_optional_int_rejects_extended_precision_numpy_fraction() -> None:
    value = np.longdouble(f"{2**53}.5")

    assert optional_int(value) is None
    assert optional_int(np.array(value)) is None
