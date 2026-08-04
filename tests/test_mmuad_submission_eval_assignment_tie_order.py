from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import match_submission_to_truth


@pytest.mark.parametrize("reverse_submission", [False, True])
@pytest.mark.parametrize("reverse_truth", [False, True])
def test_exact_time_assignment_ties_ignore_dataframe_row_order(
    reverse_submission: bool,
    reverse_truth: bool,
) -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )
    submission = truth.copy()

    if reverse_truth:
        truth = truth.iloc[::-1].reset_index(drop=True)
    if reverse_submission:
        submission = submission.iloc[::-1].reset_index(drop=True)

    matches = match_submission_to_truth(submission, truth)

    assert matches["matched"].tolist() == [True, True]
    assert matches["unmatched_reason"].tolist() == ["", ""]
    assert matches["error_3d_m"].tolist() == pytest.approx([0.0, 0.0])
