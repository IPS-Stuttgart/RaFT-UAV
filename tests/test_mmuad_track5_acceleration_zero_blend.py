from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_acceleration_limit import repair_track5_acceleration_kinks


def test_zero_blend_reports_candidate_without_applied_change() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": ["seq"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 10.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )

    repaired, diagnostics = repair_track5_acceleration_kinks(
        submission,
        max_acceleration_mps2=5.0,
        max_direct_speed_mps=20.0,
        min_interpolation_residual_m=1.0,
        iterations=2,
        repair_blend=0.0,
    )

    coordinate_columns = ["state_x_m", "state_y_m", "state_z_m"]
    assert np.array_equal(
        repaired[coordinate_columns].to_numpy(float),
        submission[coordinate_columns].to_numpy(float),
    )
    midpoint = diagnostics.loc[diagnostics["time_s"] == 1.0].iloc[0]
    assert midpoint["acceleration_limit_candidate"]
    assert not midpoint["acceleration_limit_applied"]
    assert midpoint["acceleration_limit_iteration"] == 0
    assert midpoint["acceleration_limit_displacement_m"] == 0.0
    assert not repaired["acceleration_limit_applied"].any()
