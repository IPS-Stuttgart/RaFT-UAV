from __future__ import annotations

import numpy as np

from raft_uav.mmuad.track5_jerk_limit import _row_jerk_proxy


def test_jerk_proxy_keeps_skipped_window_row_support() -> None:
    times = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    xyz = np.column_stack(
        (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0]),
            np.zeros(7),
            np.zeros(7),
        )
    )

    jerk = _row_jerk_proxy(times, xyz)

    assert np.flatnonzero(np.isfinite(jerk)).tolist() == [2, 3, 4, 5, 6]
    assert jerk[2] == 10.0
    assert np.all(jerk[3:] == 30.0)
