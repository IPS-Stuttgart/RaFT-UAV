from __future__ import annotations

import numpy as np
import pytest

from raft_uav.numeric import optional_int


@pytest.mark.parametrize(
    "value",
    [
        np.timedelta64(1, "s"),
        np.timedelta64(-5, "ms"),
        np.timedelta64("NaT"),
    ],
)
def test_optional_int_rejects_numpy_timedelta_scalars(value: np.timedelta64) -> None:
    assert optional_int(value) is None
