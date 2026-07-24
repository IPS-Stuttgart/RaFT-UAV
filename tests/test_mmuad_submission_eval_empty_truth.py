from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.evaluate import match_submission_to_truth


def test_match_submission_to_truth_reports_predictions_when_truth_is_empty() -> None:
    submission = pd.DataFrame(
        [
            {
                "sequence_id": "sequence-001",
                "time_s": 1.5,
                "track_id": "uav-7",
                "x_m": 10.0,
                "y_m": 20.0,
                "z_m": 30.0,
            }
        ]
    )
    truth = pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])

    matches = match_submission_to_truth(submission, truth)

    assert len(matches) == 1
    assert matches.loc[0, "sequence_id"] == "sequence-001"
    assert matches.loc[0, "track_id"] == "uav-7"
    assert not bool(matches.loc[0, "matched"])
    assert matches.loc[0, "unmatched_reason"] == "missing_sequence_truth"
