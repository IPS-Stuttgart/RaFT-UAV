from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_jerk_limit import repair_track5_jerk_kinks


def test_jerk_repair_rejects_duplicate_fixed_grid_keys() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 4,
            "time_s": [0.0, 1.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0, 3.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
            "Classification": [2, 2, 2, 2],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"1 duplicate \(sequence_id, time_s\) key\(s\): seq0001@1$",
    ):
        repair_track5_jerk_kinks(submission)
