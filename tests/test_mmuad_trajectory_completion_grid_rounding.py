from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    _target_times,
)


def test_inferred_grid_keeps_all_decimal_cadence_timestamps() -> None:
    group = pd.DataFrame({"time_s": [0.0, 0.1, 0.2, 0.5]})
    config = TrajectoryCompletionConfig(
        include_truth_timestamps=False,
        infer_missing_grid=True,
        max_gap_s=1.0,
    )

    target_times = _target_times(group, None, config=config)

    np.testing.assert_allclose(
        target_times,
        np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        rtol=0.0,
        atol=1.0e-12,
    )
