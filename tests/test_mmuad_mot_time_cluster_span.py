from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.mot import compute_multi_object_metrics


@pytest.mark.parametrize("include_sequence_id", [False, True])
def test_timestamp_bridge_cannot_enable_out_of_tolerance_match(
    include_sequence_id: bool,
) -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, 0.9e-9],
            "output_track_id": ["early", "bridge"],
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [1.8e-9],
            "track_id": ["truth"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    if include_sequence_id:
        estimates["sequence_id"] = "seqA"
        truth["sequence_id"] = "seqA"

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=25.0)

    assert metrics["matches"] == 0
    assert metrics["false_positive"] == 2
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
