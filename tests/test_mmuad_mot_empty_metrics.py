import pandas as pd

from raft_uav.mmuad.mot import compute_multi_object_metrics


def test_multi_object_metrics_counts_empty_predictions_as_false_negatives() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "track_id": ["uav0", "uav0"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
        }
    )

    metrics = compute_multi_object_metrics(pd.DataFrame(), truth, match_distance_m=1.0)

    assert metrics["count"] == 0
    assert metrics["gt_count"] == 2
    assert metrics["track_count"] == 0
    assert metrics["matches"] == 0
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 2
    assert metrics["id_switches"] == 0
    assert metrics["mota_like"] == 0.0
    assert metrics["motp_3d_m"] is None
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0


def test_multi_object_metrics_counts_all_invalid_predictions_as_false_negatives() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "output_track_id": ["bad"],
            "state_x_m": [float("nan")],
            "state_y_m": [0.0],
            "state_z_m": [2.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "track_id": ["uav0"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [2.0],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["count"] == 0
    assert metrics["gt_count"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["recall"] == 0.0
