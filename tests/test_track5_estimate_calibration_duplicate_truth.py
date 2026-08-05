from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_estimate_calibration import _fit_pairs


def test_fit_pairs_keeps_final_finite_duplicate_truth_snapshot() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["sequence", "sequence"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["sequence", "sequence", "sequence", "sequence"],
            "time_s": [0.0, 0.0, 0.0, 1.0],
            "x_m": [100.0, 0.0, np.nan, 1.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
        }
    )

    pairs = _fit_pairs(estimates, truth)

    assert pairs[["time_s", "truth_x_m"]].to_dict("records") == [
        {"time_s": 0.0, "truth_x_m": 0.0},
        {"time_s": 1.0, "truth_x_m": 1.0},
    ]
