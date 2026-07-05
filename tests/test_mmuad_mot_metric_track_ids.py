import pandas as pd

from raft_uav.mmuad.mot import compute_multi_object_metrics


def test_mot_metrics_accept_external_prediction_track_ids() -> None:
    truth = pd.DataFrame({"time_s": [0.0], "x_m": [0.0], "y_m": [0.0], "z_m": [0.0], "track_id": ["gt"]})
    estimate = pd.DataFrame({"time_s": [0.0], "state_x_m": [0.0], "state_y_m": [0.0], "state_z_m": [0.0], "track_id": ["pred"]})

    metrics = compute_multi_object_metrics(estimate, truth, match_distance_m=1.0)

    assert metrics["matches"] == 1
    assert metrics["track_count"] == 1
    assert metrics["id_switches"] == 0
