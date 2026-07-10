from __future__ import annotations

import numpy as np

from raft_uav.mmuad.track5_jerk_limit import _row_jerk_proxy


def test_jerk_proxy_uses_actual_rows_after_skipped_window() -> None:
    times = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    xyz = np.column_stack(
        [
            np.array([0.0, 0.0, 1.0, 8.0, 27.0]),
            np.zeros(5),
            np.zeros(5),
        ]
    )

    jerk = _row_jerk_proxy(times, xyz)

    assert np.isnan(jerk[0])
    assert np.allclose(jerk[1:], 6.0)
