import pandas as pd

from raft_uav.mmuad.mot import compute_multi_object_metrics


def test_pooled_mot_track_count_scopes_prediction_ids_by_sequence() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["sequence-a", "sequence-b"],
            "time_s": [0.0, 0.0],
            "state_x_m": [0.0, 0.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "output_track_id": ["mot_1", "mot_1"],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["sequence-a", "sequence-b"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "track_id": ["truth_1", "truth_1"],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["matches"] == 2
    assert metrics["track_count"] == 2
    assert metrics["id_switches"] == 0
