from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_jerk_limit import repair_track5_jerk_kinks


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_x_m": [0.0, 1.0, 30.0, 3.0, 4.0, 5.0],
            "state_y_m": [0.0] * 6,
            "state_z_m": [1.0] * 6,
            "Classification": [2] * 6,
        }
    )


def test_zero_blend_keeps_jerk_candidates_diagnostic_only() -> None:
    original = _submission()
    repaired, diagnostics = repair_track5_jerk_kinks(
        original,
        max_jerk_mps3=5.0,
        smoothness_weight=100.0,
        min_correction_m=1.0,
        iterations=3,
        repair_blend=0.0,
    )

    coordinate_columns = ["state_x_m", "state_y_m", "state_z_m"]
    assert np.allclose(
        repaired[coordinate_columns].to_numpy(float),
        original[coordinate_columns].to_numpy(float),
    )
    assert diagnostics["jerk_limit_mps3"].gt(5.0).any()
    assert not diagnostics["jerk_limit_applied"].any()
    assert np.allclose(diagnostics["jerk_limit_displacement_m"].to_numpy(float), 0.0)
