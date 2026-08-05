from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import match_submission_to_truth, metrics_from_matches


def _anonymous_truth(*, count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * count,
            "time_s": [0.0] * count,
            "track_id": [""] * count,
            "x_m": [float(10 * index) for index in range(count)],
            "y_m": [0.0] * count,
            "z_m": [0.0] * count,
        }
    )


def test_coverage_counts_coincident_anonymous_truth_rows() -> None:
    truth = _anonymous_truth(count=2)
    submission = truth.copy()

    matches = match_submission_to_truth(submission, truth)
    result = metrics_from_matches(matches, submission=submission, truth=truth)

    assert result["pooled"]["matched_count"] == 2
    assert result["pooled"]["covered_truth_count"] == 2
    assert result["pooled"]["truth_coverage_fraction"] == pytest.approx(1.0)
    assert result["sequences"]["seqA"]["covered_truth_count"] == 2
    assert result["sequences"]["seqA"]["truth_coverage_fraction"] == pytest.approx(1.0)


def test_coverage_caps_duplicate_matches_at_truth_multiplicity() -> None:
    truth = _anonymous_truth(count=1)
    submission = pd.concat([truth, truth], ignore_index=True)
    matches = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "track_id": ["uav0", "uav0"],
            "truth_time_s": [0.0, 0.0],
            "truth_track_id": ["", ""],
            "time_delta_s": [0.0, 0.0],
            "matched": [True, True],
            "unmatched_reason": ["", ""],
            "error_2d_m": [0.0, 0.0],
            "error_3d_m": [0.0, 0.0],
            "vertical_error_m": [0.0, 0.0],
        }
    )

    result = metrics_from_matches(matches, submission=submission, truth=truth)

    assert result["pooled"]["matched_count"] == 2
    assert result["pooled"]["covered_truth_count"] == 1
    assert result["pooled"]["truth_coverage_fraction"] == pytest.approx(1.0)
