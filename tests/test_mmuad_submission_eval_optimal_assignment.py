from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.evaluate import match_submission_to_truth


def test_submission_matching_maximizes_valid_cardinality_before_time_delta() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 2.0],
            "x_m": [0.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )
    submission = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [1.0, 0.0],
            "track_id": ["first", "second"],
            "x_m": [2.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    matches = match_submission_to_truth(submission, truth, max_time_delta_s=1.1)

    assert matches["matched"].tolist() == [True, True]
    assert matches["truth_time_s"].tolist() == [2.0, 0.0]
    assert matches["unmatched_reason"].tolist() == ["", ""]
