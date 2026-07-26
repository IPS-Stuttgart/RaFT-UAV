from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_jerk_limit import repair_track5_jerk_kinks


def test_multistep_jerk_diagnostics_report_net_displacement() -> None:
    original = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_x_m": [1.0, 1.0, 13.0, -13.0, -23.0, -7.0],
            "state_y_m": [-1.0, -5.0, 9.0, -6.0, -2.0, -5.0],
            "state_z_m": [0.0] * 6,
            "Classification": [2] * 6,
        }
    )

    repaired, diagnostics = repair_track5_jerk_kinks(
        original,
        max_jerk_mps3=0.1,
        smoothness_weight=10.0,
        min_correction_m=0.0,
        iterations=3,
        repair_blend=0.5,
    )

    coordinate_columns = ["state_x_m", "state_y_m", "state_z_m"]
    actual_displacement = np.linalg.norm(
        repaired[coordinate_columns].to_numpy(float)
        - original[coordinate_columns].to_numpy(float),
        axis=1,
    )
    reported_displacement = diagnostics["jerk_limit_displacement_m"].to_numpy(float)

    assert diagnostics["jerk_limit_applied"].any()
    np.testing.assert_allclose(reported_displacement, actual_displacement)
