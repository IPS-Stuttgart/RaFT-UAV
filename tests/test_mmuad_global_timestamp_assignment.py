from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import evaluate_mmaud_results
from raft_uav.mmuad.submission import validate_official_track5_submission
from raft_uav.mmuad.timestamp_assignment import optimal_timestamp_assignment


def _overlapping_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [-0.05, 0.04],
            "Position": ["(0,0,0)", "(1,0,0)"],
            "Classification": [1, 1],
        }
    )


def _overlapping_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 0.06],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "uav_type": ["1", "1"],
        }
    )


def test_equal_error_assignment_preserves_stable_request_order() -> None:
    assignment = optimal_timestamp_assignment(
        [0.0, 0.04],
        [0.02],
        tolerance_s=0.05,
    )

    assert assignment == {0: 0}


def test_stable_tie_break_never_increases_timestamp_error() -> None:
    assignment = optimal_timestamp_assignment(
        [0.0, 1.0e-16],
        [-1.0e-16, 0.0, 1.0e-16],
        tolerance_s=1.0,
    )

    assert assignment == {0: 1, 1: 2}


def test_assignment_rejects_rounded_search_bound_outside_tolerance() -> None:
    gap = abs(0.04 - 0.03)
    assert gap > 0.01

    assignment = optimal_timestamp_assignment(
        [0.03],
        [0.04],
        tolerance_s=0.01,
    )

    assert assignment == {}


def test_assignment_keeps_rounded_search_bound_inside_tolerance() -> None:
    gap = abs(0.04 - 0.01)
    assert gap <= 0.03

    assignment = optimal_timestamp_assignment(
        [0.04],
        [0.01],
        tolerance_s=0.03,
    )

    assert assignment == {0: 0}


def test_public_track5_matching_uses_global_one_to_one_assignment() -> None:
    evaluated = evaluate_mmaud_results(
        _overlapping_results(),
        _overlapping_truth(),
        metric_protocol="public-track5",
        timestamp_tolerance_s=0.06,
    )

    summary = evaluated["summary"]
    assert summary["matched_count"] == 2
    assert summary["missing_prediction_count"] == 0
    assert summary["extra_prediction_count"] == 0
    assert summary["leaderboard_ready"] is True
    assert summary["pooled"]["mean_square_loss_m2"] == pytest.approx(0.0)
    assert evaluated["rows"]["time_delta_s"].tolist() == pytest.approx([-0.05, -0.02])


def test_submission_preflight_uses_global_one_to_one_assignment(tmp_path) -> None:
    results_path = tmp_path / "mmaud_results.csv"
    _overlapping_results().to_csv(results_path, index=False)
    template = _overlapping_truth().rename(
        columns={"sequence_id": "Sequence", "time_s": "Timestamp"}
    )[["Sequence", "Timestamp"]]

    validation = validate_official_track5_submission(
        results_path,
        template=template,
        timestamp_tolerance_s=0.06,
        require_zip=False,
    )

    assert validation.summary["missing_template_timestamp_count"] == 0
    assert validation.summary["extra_prediction_count"] == 0
    assert validation.summary["score_valid_for_leaderboard"] is True
    template_rows = validation.rows.loc[validation.rows["row_type"] == "template"]
    assert template_rows["status"].tolist() == [
        "covered_template_timestamp",
        "covered_template_timestamp",
    ]
