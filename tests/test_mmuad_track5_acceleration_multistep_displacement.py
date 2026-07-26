from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_acceleration_limit import (
    repair_track5_acceleration_kinks,
)


def test_multistep_acceleration_diagnostics_report_net_displacement() -> None:
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

    repaired, diagnostics = repair_track5_acceleration_kinks(
        original,
        max_acceleration_mps2=0.1,
        max_direct_speed_mps=100.0,
        min_interpolation_residual_m=0.0,
        iterations=3,
        repair_blend=0.5,
    )

    coordinate_columns = ["state_x_m", "state_y_m", "state_z_m"]
    actual_displacement = np.linalg.norm(
        repaired[coordinate_columns].to_numpy(float)
        - original[coordinate_columns].to_numpy(float),
        axis=1,
    )
    diagnostic_displacement = diagnostics[
        "acceleration_limit_displacement_m"
    ].to_numpy(float)
    repaired_displacement = repaired[
        "acceleration_limit_displacement_m"
    ].to_numpy(float)

    assert diagnostics["acceleration_limit_applied"].any()
    np.testing.assert_allclose(diagnostic_displacement, actual_displacement)
    np.testing.assert_allclose(repaired_displacement, actual_displacement)
