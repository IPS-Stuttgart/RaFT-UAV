from __future__ import annotations

from decimal import Decimal

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
