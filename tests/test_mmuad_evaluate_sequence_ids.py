from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.evaluate import match_submission_to_truth


def test_match_submission_to_truth_normalizes_direct_frame_sequence_ids() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": [1],
            "time_s": [0.0],
            "track_id": ["uav0"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["1"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )

    matches = match_submission_to_truth(submission, truth)

    assert len(matches) == 1
    assert bool(matches.loc[0, "matched"])
    assert matches.loc[0, "unmatched_reason"] == ""
    assert matches.loc[0, "error_3d_m"] == 0.0


def test_match_submission_to_truth_strips_direct_frame_sequence_ids() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": [" seq-01 "],
            "time_s": [0.0],
            "track_id": ["uav0"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq-01"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )

    matches = match_submission_to_truth(submission, truth)

    assert len(matches) == 1
    assert bool(matches.loc[0, "matched"])
    assert matches.loc[0, "sequence_id"] == "seq-01"
    assert matches.loc[0, "unmatched_reason"] == ""
    assert matches.loc[0, "error_3d_m"] == 0.0
