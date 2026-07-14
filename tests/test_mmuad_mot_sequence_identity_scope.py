from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.mot import compute_multi_object_metrics


def test_pooled_mot_metrics_do_not_cross_sequence_identity_boundaries() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "output_track_id": ["prediction-A", "prediction-B"],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "track_id": ["shared", "shared"],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["matches"] == 2
    assert metrics["id_switches"] == 0
    assert metrics["mota_like"] == 1.0


def test_mot_identity_scoping_accepts_pandas_string_track_ids() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "output_track_id": ["prediction-A", "prediction-B"],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": pd.Series(["seqA", "seqB"], dtype="string"),
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "track_id": pd.Series(["shared", "shared"], dtype="string"),
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["matches"] == 2
    assert metrics["id_switches"] == 0
    assert metrics["mota_like"] == 1.0


def test_mot_metrics_still_count_switches_within_one_sequence() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "output_track_id": ["prediction-A", "prediction-B"],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "track_id": ["shared", "shared"],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["matches"] == 2
    assert metrics["id_switches"] == 1
    assert metrics["mota_like"] == 0.5
