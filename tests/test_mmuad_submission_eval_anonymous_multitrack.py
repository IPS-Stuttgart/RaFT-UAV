from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import match_submission_to_truth


def _multi_track_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 0.0],
            "track_id": ["uav0", "uav1"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


@pytest.mark.parametrize("track_ids", [None, ["", ""]])
def test_match_rejects_anonymous_predictions_for_multi_track_truth(
    track_ids: list[str] | None,
) -> None:
    rows: dict[str, list[object]] = {
        "sequence_id": ["s1", "s1"],
        "time_s": [0.0, 0.0],
        "x_m": [0.0, 10.0],
        "y_m": [0.0, 0.0],
        "z_m": [0.0, 0.0],
    }
    if track_ids is not None:
        rows["track_id"] = track_ids

    matches = match_submission_to_truth(pd.DataFrame(rows), _multi_track_truth())

    assert matches["matched"].tolist() == [False, False]
    assert matches["unmatched_reason"].tolist() == [
        "track_id_mismatch",
        "track_id_mismatch",
    ]


def test_anonymous_prediction_keeps_single_track_fallback() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "time_s": [0.0],
            "track_id": ["actual-id"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )

    matches = match_submission_to_truth(submission, truth)

    assert matches["matched"].tolist() == [True]
