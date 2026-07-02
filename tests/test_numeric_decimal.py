from __future__ import annotations

from decimal import Decimal

from raft_uav.numeric import optional_int


def test_optional_int_rejects_fractional_decimal_text_exactly() -> None:
    fractional = "12345.5"

    assert optional_int(fractional) is None
    assert optional_int(Decimal(fractional)) is None
