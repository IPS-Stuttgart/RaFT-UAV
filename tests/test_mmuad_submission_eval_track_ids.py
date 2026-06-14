from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.evaluate import match_submission_to_truth, metrics_from_matches


def test_submission_evaluator_rejects_wrong_track_id_when_ids_overlap() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 0.0],
            "track_id": ["uav_a", "wrong_id"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
            "score": [1.0, 1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 0.0],
            "track_id": ["uav_a", "uav_b"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
        }
    )

    matches = match_submission_to_truth(submission, truth)

    by_track = {row["track_id"]: row for _, row in matches.iterrows()}
    assert bool(by_track["uav_a"]["matched"])
    assert not bool(by_track["wrong_id"]["matched"])
    assert by_track["wrong_id"]["unmatched_reason"] == "track_id_mismatch"

    metrics = metrics_from_matches(matches, submission=submission, truth=truth)
    assert metrics["pooled"]["matched_count"] == 1
    assert metrics["pooled"]["unmatched_prediction_count"] == 1
    assert metrics["pooled"]["covered_truth_count"] == 1
