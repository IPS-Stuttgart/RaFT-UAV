from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.mot import compute_multi_object_metrics


def test_mot_matching_maximizes_valid_cardinality_before_distance() -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "state_x_m": [1.0, -2.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "output_track_id": ["pred_a", "pred_b"],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 3.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "track_id": ["truth_a", "truth_b"],
        }
    )

    metrics = compute_multi_object_metrics(
        estimates,
        truth,
        match_distance_m=2.1,
    )

    assert metrics["matches"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert np.isclose(metrics["motp_3d_m"], 2.0)
