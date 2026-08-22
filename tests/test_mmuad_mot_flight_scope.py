import pandas as pd

from raft_uav.mmuad.mot import (
    compute_multi_object_metrics,
    run_mmuad_multi_object_tracker,
)
from raft_uav.mmuad.schema import CandidateFrame


def test_mot_metrics_scope_reused_truth_identity_by_flight():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight_a", "flight_b"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "output_track_id": ["mot_1", "mot_2"],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight_a", "flight_b"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "track_id": ["target", "target"],
        }
    )

    metrics = compute_multi_object_metrics(
        estimates,
        truth,
        match_distance_m=1.0,
    )

    assert metrics["matches"] == 2
    assert metrics["id_switches"] == 0


def test_mot_tracker_restarts_state_for_each_flight_alias():
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

    assert result.metrics["pooled"]["track_count"] == 2
    assert set(result.estimates["flight_id"]) == {"flight_a", "flight_b"}
    assert result.estimates["sequence_id"].tolist() == ["shared", "shared"]
    assert len(result.metrics["sequences"]) == 2
