from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.evaluate import match_submission_to_truth


def test_match_submission_to_truth_ignores_blank_track_ids_when_restricted() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 0.0],
            "track_id": ["uav1", ""],
            "x_m": [10.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 0.0],
            "track_id": ["uav1", ""],
            "x_m": [10.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    matches = match_submission_to_truth(submission, truth)

    assert bool(matches.loc[0, "matched"])
    assert matches.loc[0, "truth_track_id"] == "uav1"
    assert bool(matches.loc[1, "matched"]) is False
    assert matches.loc[1, "unmatched_reason"] == "track_id_mismatch"
