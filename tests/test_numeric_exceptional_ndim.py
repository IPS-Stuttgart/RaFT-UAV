from __future__ import annotations

from raft_uav.numeric import optional_float, optional_int


class _ExplodingArrayLike:
    @property
    def ndim(self) -> object:
        raise RuntimeError("broken array metadata")

    def __float__(self) -> float:
        return 4.0


def test_optional_numeric_rejects_exceptional_array_dimensionality() -> None:
    value = _ExplodingArrayLike()

    assert optional_float(value) is None
    assert optional_int(value) is None
