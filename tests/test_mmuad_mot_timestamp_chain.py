from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.mot import compute_multi_object_metrics


def test_timestamp_chain_does_not_match_rows_beyond_tolerance() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.75e-9],
            "output_track_id": ["pred_early", "pred_bridge"],
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.75e-9, 1.5e-9],
            "track_id": ["truth_bridge", "truth_late"],
            "x_m": [100.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    metrics = compute_multi_object_metrics(
        estimates,
        truth,
        match_distance_m=1.0,
    )

    assert metrics["matches"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["motp_3d_m"] == 0.0
