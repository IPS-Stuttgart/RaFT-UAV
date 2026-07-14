from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.mot import compute_multi_object_metrics


def test_adjacent_large_timestamps_stay_separate() -> None:
    t0 = 1700000000.0
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [t0],
            "output_track_id": ["pred1"],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [t0, t0 + 1.0],
            "track_id": ["truth1", "truth2"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=25.0)

    assert metrics["matches"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


@pytest.mark.parametrize("include_sequence_id", [False, True])
def test_near_equal_timestamps_are_counted_once(include_sequence_id: bool) -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [1.0],
            "output_track_id": ["pred1"],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [1.0 + 5.0e-10],
            "track_id": ["truth1"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    if include_sequence_id:
        estimates["sequence_id"] = "seqA"
        truth["sequence_id"] = "seqA"

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=25.0)

    assert metrics["matches"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
