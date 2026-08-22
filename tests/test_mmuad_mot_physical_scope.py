import pandas as pd
import pytest

from raft_uav.mmuad.mot import (
    compute_multi_object_metrics,
    run_mmuad_multi_object_tracker,
)
from raft_uav.mmuad.schema import CandidateFrame


def _estimate_rows(**overrides):
    rows = {
        "sequence_id": ["shared", "shared"],
        "time_s": [0.0, 1.0],
        "state_x_m": [0.0, 100.0],
        "state_y_m": [0.0, 0.0],
        "state_z_m": [0.0, 0.0],
        "output_track_id": ["mot_1", "mot_2"],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def _truth_rows(**overrides):
    rows = {
        "sequence_id": ["shared", "shared"],
        "time_s": [0.0, 1.0],
        "x_m": [0.0, 100.0],
        "y_m": [0.0, 0.0],
        "z_m": [0.0, 0.0],
        "track_id": ["target", "target"],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_mot_metrics_scope_reused_truth_identity_by_flight():
    estimates = _estimate_rows(flight_id=["flight_a", "flight_b"])
    truth = _truth_rows(flight_id=["flight_a", "flight_b"])

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["matches"] == 2
    assert metrics["id_switches"] == 0
    assert metrics["track_count"] == 2


def test_mot_metrics_never_cross_match_same_time_flights():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight_a", "flight_b"],
            "time_s": [0.0, 0.0],
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "output_track_id": ["mot_1", "mot_1"],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight_a", "flight_b"],
            "time_s": [0.0, 0.0],
            "x_m": [100.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "track_id": ["target", "target"],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["matches"] == 0
    assert metrics["false_positive"] == 2
    assert metrics["false_negative"] == 2


def test_mot_metrics_scope_reused_ids_by_sequence_without_flight_metadata():
    estimates = _estimate_rows(
        sequence_id=["sequence_a", "sequence_b"],
        output_track_id=["mot_1", "mot_1"],
    )
    truth = _truth_rows(
        sequence_id=["sequence_a", "sequence_b"],
        track_id=["target", "target"],
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["matches"] == 2
    assert metrics["track_count"] == 2
    assert metrics["id_switches"] == 0


def test_mot_metrics_truth_only_scope_does_not_require_estimate_metadata():
    truth = _truth_rows(
        sequence_id=["sequence_a", "sequence_b"],
        flight_id=["flight_a", "flight_b"],
    )

    metrics = compute_multi_object_metrics(pd.DataFrame(), truth, match_distance_m=1.0)

    assert metrics["gt_count"] == 2
    assert metrics["false_negative"] == 2
    assert metrics["id_switches"] == 0


def test_mot_metrics_reject_ambiguous_one_sided_flight_metadata():
    estimates = _estimate_rows(flight_id=["flight_a", "flight_b"])
    truth = _truth_rows()

    with pytest.raises(ValueError, match="both carry flight_id"):
        compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)


def test_mot_tracker_restarts_state_for_each_physical_flight():
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["shared", "shared"],
                "flight_id": ["flight_a", "flight_b"],
                "time_s": [0.0, 1.0],
                "source": ["radar", "radar"],
                "x_m": [0.0, 0.0],
                "y_m": [0.0, 0.0],
                "z_m": [0.0, 0.0],
                "confidence": [1.0, 1.0],
            }
        )
    )

    result = run_mmuad_multi_object_tracker(candidates)

    assert result.estimates["update_action"].tolist() == ["new_track", "new_track"]
    assert result.metrics["pooled"]["track_count"] == 2
    assert set(result.estimates["flight_id"]) == {"flight_a", "flight_b"}
    assert result.estimates["sequence_id"].tolist() == ["shared", "shared"]
    assert set(result.selected_tracklets["flight_id"]) == {"flight_a", "flight_b"}
    assert len(result.metrics["sequences"]) == 2
