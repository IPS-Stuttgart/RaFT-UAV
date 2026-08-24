from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import _mean_final_error, match_submission_to_truth


def test_submission_fde_rejects_partially_missing_flight_ids() -> None:
    matches = pd.DataFrame(
        {
            "sequence_id": ["shared"] * 4,
            "flight_id": ["flight-a", None, "flight-b", "flight-b"],
            "truth_track_id": ["uav0"] * 4,
            "truth_time_s": [0.0, 1.0, 0.0, 1.0],
            "error_3d_m": [1.0, 2.0, 10.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match="flight_id metadata is partially missing"):
        _mean_final_error(matches, "error_3d_m")


def test_empty_truth_matching_rejects_partially_missing_flight_ids() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", None],
            "time_s": [0.0, 0.0],
            "track_id": ["uav0", "uav0"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        columns=["sequence_id", "time_s", "track_id", "x_m", "y_m", "z_m"]
    )

    with pytest.raises(ValueError, match="flight_id metadata is partially missing"):
        match_submission_to_truth(submission, truth)


def test_submission_fde_preserves_all_missing_flight_id_fallback() -> None:
    matches = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": [None, None],
            "truth_track_id": ["uav0", "uav0"],
            "truth_time_s": [0.0, 1.0],
            "error_3d_m": [1.0, 2.0],
        }
    )

    assert _mean_final_error(matches, "error_3d_m") == 2.0
