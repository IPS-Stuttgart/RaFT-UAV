from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import _mean_final_error, match_submission_to_truth


def _pooled_flight_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "track_id": ["uav0", "uav0"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


def test_submission_matching_uses_joint_sequence_and_flight_scope() -> None:
    truth = _pooled_flight_rows()
    submission = _pooled_flight_rows()

    matches = match_submission_to_truth(submission, truth)
    matches = matches.sort_values("flight_id", kind="mergesort").reset_index(drop=True)

    assert matches["flight_id"].tolist() == ["flight-a", "flight-b"]
    assert matches["matched"].tolist() == [True, True]
    assert matches["error_3d_m"].tolist() == [0.0, 0.0]
    assert matches["unmatched_reason"].tolist() == ["", ""]


def test_submission_fde_is_scoped_by_physical_flight() -> None:
    matches = pd.DataFrame(
        {
            "sequence_id": ["shared"] * 4,
            "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "truth_track_id": ["uav0"] * 4,
            "truth_time_s": [0.0, 1.0, 0.0, 1.0],
            "error_3d_m": [1.0, 2.0, 10.0, 20.0],
        }
    )

    assert _mean_final_error(matches, "error_3d_m") == 11.0


def test_submission_matching_rejects_ambiguous_one_sided_flight_metadata() -> None:
    submission = _pooled_flight_rows()
    truth = _pooled_flight_rows().drop(columns=["flight_id"])

    with pytest.raises(
        ValueError,
        match="without matching flight_id metadata",
    ):
        match_submission_to_truth(submission, truth)
