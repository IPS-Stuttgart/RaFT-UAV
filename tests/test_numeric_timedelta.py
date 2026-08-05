from __future__ import annotations

import numpy as np

from raft_uav.numeric import optional_int


def test_optional_int_rejects_numpy_timedelta_scalars() -> None:
    values = (
        np.timedelta64(5, "s"),
        np.timedelta64("NaT"),
        np.array(np.timedelta64(5, "s"), dtype=object),
    )

    for value in values:
        assert optional_int(value) is None


def test_optional_int_preserves_numpy_integer_scalars() -> None:
    assert optional_int(np.int64(7)) == 7
