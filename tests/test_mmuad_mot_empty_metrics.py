import pandas as pd

from raft_uav.mmuad.mot import compute_multi_object_metrics, run_mmuad_multi_object_tracker
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame


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


def test_multi_object_tracker_empty_candidates_counts_truth_false_negatives_by_sequence() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(columns=["sequence_id", "time_s", "source", "x_m", "y_m", "z_m"])
    )
    truth = TruthFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqB"],
                "time_s": [0.0, 0.0],
                "track_id": ["uav0", "uav1"],
                "x_m": [0.0, 10.0],
                "y_m": [0.0, 0.0],
                "z_m": [2.0, 2.0],
            }
        )
    )

    output = run_mmuad_multi_object_tracker(candidates, truth)

    assert output.estimates.empty
    assert output.selected_tracklets.empty
    assert output.metrics["pooled"]["gt_count"] == 2
    assert output.metrics["pooled"]["false_negative"] == 2
    assert output.metrics["pooled"]["recall"] == 0.0
    assert set(output.metrics["sequences"]) == {"seqA", "seqB"}
    assert output.metrics["sequences"]["seqA"]["false_negative"] == 1
    assert output.metrics["sequences"]["seqB"]["false_negative"] == 1


def test_multi_object_tracker_keeps_truth_only_sequence_metrics() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seqA"],
                "time_s": [0.0],
                "source": ["radar"],
                "x_m": [0.0],
                "y_m": [0.0],
                "z_m": [2.0],
                "confidence": [1.0],
            }
        )
    )
    truth = TruthFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqB"],
                "time_s": [0.0, 0.0],
                "track_id": ["uav0", "uav1"],
                "x_m": [0.0, 10.0],
                "y_m": [0.0, 0.0],
                "z_m": [2.0, 2.0],
            }
        )
    )

    output = run_mmuad_multi_object_tracker(candidates, truth)

    assert set(output.metrics["sequences"]) == {"seqA", "seqB"}
    assert output.metrics["sequences"]["seqA"]["matches"] == 1
    assert output.metrics["sequences"]["seqB"]["false_negative"] == 1
    assert output.metrics["pooled"]["gt_count"] == 2
    assert output.metrics["pooled"]["matches"] == 1
    assert output.metrics["pooled"]["false_negative"] == 1
