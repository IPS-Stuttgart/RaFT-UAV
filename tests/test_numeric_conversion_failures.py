from __future__ import annotations

from raft_uav.numeric import optional_float, optional_int


class _BrokenFloatConversion:
    def __float__(self) -> float:
        return (_ for _ in ()).throw(RuntimeError("synthetic conversion failure"))


def test_optional_numeric_rejects_scalar_values_with_broken_float_conversion() -> None:
    value = _BrokenFloatConversion()

    assert optional_float(value) is None
    assert optional_int(value) is None
