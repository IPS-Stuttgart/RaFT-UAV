from __future__ import annotations

import numpy as np

from raft_uav.numeric import optional_float, optional_int


def test_optional_float_rejects_masked_scalar_values() -> None:
    assert optional_float(np.ma.masked) is None
    assert optional_float(np.ma.array(1.0, mask=True)) is None


def test_optional_int_rejects_masked_scalar_values() -> None:
    assert optional_int(np.ma.masked) is None
    assert optional_int(np.ma.array(4, mask=True)) is None


def test_optional_numeric_accepts_unmasked_scalar_values() -> None:
    assert optional_float(np.ma.array(2.5, mask=False)) == 2.5
    assert optional_int(np.ma.array(9, mask=False)) == 9
