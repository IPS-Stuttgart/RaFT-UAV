from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.mot import compute_multi_object_metrics


@pytest.mark.parametrize("include_sequence_id", [False, True])
def test_tolerance_matched_timestamps_are_counted_once(
    include_sequence_id: bool,
) -> None:
    estimate_data: dict[str, list[object]] = {
        "time_s": [1.0],
        "output_track_id": ["pred1"],
        "state_x_m": [0.0],
        "state_y_m": [0.0],
        "state_z_m": [0.0],
    }
    truth_data: dict[str, list[object]] = {
        "time_s": [1.0 + 5.0e-10],
        "track_id": ["truth1"],
        "x_m": [0.0],
        "y_m": [0.0],
        "z_m": [0.0],
    }
    if include_sequence_id:
        estimate_data["sequence_id"] = ["seqA"]
        truth_data["sequence_id"] = ["seqA"]

    metrics = compute_multi_object_metrics(
        pd.DataFrame(estimate_data),
        pd.DataFrame(truth_data),
        match_distance_m=25.0,
    )

    assert metrics["matches"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
