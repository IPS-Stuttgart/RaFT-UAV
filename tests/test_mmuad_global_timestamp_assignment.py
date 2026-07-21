from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import evaluate_mmaud_results
from raft_uav.mmuad.submission import validate_official_track5_submission


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
