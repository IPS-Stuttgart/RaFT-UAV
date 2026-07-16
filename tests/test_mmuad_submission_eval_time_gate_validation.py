from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import (
    _optimal_time_assignment,
    match_submission_to_truth,
)


def _submission_and_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    submission = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "track_id": ["trackA"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [10.0],
            "track_id": ["trackA"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    return submission, truth


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.array([0.5]),
        np.array([True]),
        -0.1,
        float("nan"),
        float("inf"),
        0.5 + 0.0j,
        np.ma.masked,
    ],
)
def test_submission_matching_rejects_invalid_time_gates(value: object) -> None:
    submission, truth = _submission_and_truth()

    with pytest.raises(
        ValueError,
        match="max_time_delta_s must be a finite nonnegative real scalar",
    ):
        match_submission_to_truth(
            submission,
            truth,
            max_time_delta_s=value,
        )


def test_submission_matching_validates_time_gate_before_empty_return() -> None:
    submission, truth = _submission_and_truth()

    with pytest.raises(ValueError, match="max_time_delta_s"):
        match_submission_to_truth(
            submission.iloc[0:0],
            truth,
            max_time_delta_s=True,
        )


def test_optimal_time_assignment_validates_direct_calls() -> None:
    submission, truth = _submission_and_truth()

    with pytest.raises(ValueError, match="max_time_delta_s"):
        _optimal_time_assignment(
            submission,
            truth,
            restrict_to_track_id=True,
            max_time_delta_s=np.array([0.5]),
        )


@pytest.mark.parametrize("value", [np.float64(10.0), np.array(10.0)])
def test_submission_matching_accepts_real_scalar_like_time_gates(value: object) -> None:
    submission, truth = _submission_and_truth()

    matches = match_submission_to_truth(
        submission,
        truth,
        max_time_delta_s=value,
    )

    assert matches["matched"].tolist() == [True]
    assert matches["time_delta_s"].tolist() == [10.0]
