from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import evaluate_mmaud_results


def _one_exact_prediction() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "timestamp": [0.0],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "uav_type": ["1"],
            "score": [1.0],
        }
    )


def _duplicate_truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": ["0", 0.0],
            "x_m": [100.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "uav_type": ["1", "1"],
        }
    )


def test_public_track5_evaluator_keeps_final_duplicate_truth_row() -> None:
    evaluated = evaluate_mmaud_results(
        _one_exact_prediction(),
        _duplicate_truth_rows(),
        metric_protocol="public-track5",
        timestamp_tolerance_s=0.0,
    )

    summary = evaluated["summary"]
    assert summary["truth_count"] == 1
    assert summary["matched_count"] == 1
    assert summary["missing_prediction_count"] == 0
    assert summary["extra_prediction_count"] == 0
    assert summary["duplicate_prediction_count"] == 0
    assert summary["leaderboard_ready"] is True
    assert summary["pooled"]["mean_square_loss_m2"] == pytest.approx(0.0)
    assert evaluated["rows"]["truth_x_m"].tolist() == [0.0]


def test_nearest_time_evaluator_keeps_final_duplicate_truth_row() -> None:
    evaluated = evaluate_mmaud_results(
        _one_exact_prediction(),
        _duplicate_truth_rows(),
        metric_protocol="nearest-time",
        max_time_delta_s=0.0,
    )

    summary = evaluated["summary"]
    assert summary["matched_count"] == 1
    assert summary["unmatched_count"] == 0
    assert summary["pooled"]["mean_square_loss_m2"] == pytest.approx(0.0)
    assert evaluated["rows"]["truth_x_m"].tolist() == [0.0]
