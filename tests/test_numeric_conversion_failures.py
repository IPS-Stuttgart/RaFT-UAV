from __future__ import annotations

import numpy as np

from raft_uav.numeric import optional_float, optional_int


def test_optional_numeric_rejects_extended_precision_values_that_do_not_fit_float() -> None:
    value = np.longdouble("1e10000")

    assert optional_float(value) is None
    assert optional_int(value) is None
