from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.completion import (
    complete_results_to_truth_timestamps,
    completion_summary,
)
from raft_uav.mmuad.submission import UG2_RESULT_COLUMNS


def _truth_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
        }
    )


def _interpolation_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "timestamp": [0.0, 10.0],
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "z": [2.0, 2.0],
            "uav_type": ["2", "2"],
            "score": [1.0, 1.0],
        }
    )


def test_completion_reports_missing_predictions_for_empty_result_table() -> None:
    results = pd.DataFrame(columns=UG2_RESULT_COLUMNS)

    completed = complete_results_to_truth_timestamps(results, _truth_template())

    assert completed.rows.empty
    assert completed.diagnostics["completion_method"].tolist() == [
        "missing_sequence_prediction",
        "missing_sequence_prediction",
    ]
    summary = completion_summary(completed)
    assert summary["requested_count"] == 2
    assert summary["completed_count"] == 0
    assert summary["dropped_count"] == 2
    assert summary["sequences"]["seq1"] == {
        "requested_count": 2,
        "completed_count": 0,
        "dropped_count": 2,
        "completion_method_counts": {"missing_sequence_prediction": 2},
        "completion_coverage_fraction": 0.0,
        "all_requested_timestamps_completed": False,
    }


def test_completion_treats_all_nonfinite_result_rows_as_missing_predictions() -> None:
    results = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "timestamp": [np.nan],
            "x": [1.0],
            "y": [0.0],
            "z": [2.0],
            "uav_type": ["Mavic3"],
            "score": [1.0],
        }
    )

    completed = complete_results_to_truth_timestamps(results, _truth_template())

    assert completed.rows.empty
    assert completed.diagnostics["completion_method"].tolist() == [
        "missing_sequence_prediction",
        "missing_sequence_prediction",
    ]


@pytest.mark.parametrize("bad_gap", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_completion_rejects_invalid_interpolation_gap_threshold(bad_gap: float) -> None:
    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": [5.0]})

    with pytest.raises(ValueError, match="max_interpolation_gap_s"):
        complete_results_to_truth_timestamps(
            _interpolation_results(),
            template,
            max_interpolation_gap_s=bad_gap,
        )


def test_completion_accepts_zero_interpolation_gap_threshold() -> None:
    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": [5.0]})

    completed = complete_results_to_truth_timestamps(
        _interpolation_results(),
        template,
        max_interpolation_gap_s=0.0,
    )

    assert completed.rows.iloc[0]["x"] == 0.0
    assert completed.diagnostics["completion_method"].tolist() == ["hold_before"]


def test_completion_summary_reports_sequence_readiness() -> None:
    results = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "timestamp": [0.0],
            "x": [1.0],
            "y": [0.0],
            "z": [2.0],
            "uav_type": ["2"],
            "score": [1.0],
        }
    )
    template = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq2"],
            "time_s": [0.0, 1.0, 0.0],
        }
    )

    completed = complete_results_to_truth_timestamps(results, template)
    summary = completion_summary(completed)

    assert summary["requested_count"] == 3
    assert summary["completed_count"] == 2
    assert summary["dropped_count"] == 1
    assert summary["sequences"]["seq1"] == {
        "requested_count": 2,
        "completed_count": 2,
        "dropped_count": 0,
        "completion_method_counts": {"exact": 1, "hold_single": 1},
        "completion_coverage_fraction": 1.0,
        "all_requested_timestamps_completed": True,
    }
    assert summary["sequences"]["seq2"] == {
        "requested_count": 1,
        "completed_count": 0,
        "dropped_count": 1,
        "completion_method_counts": {"missing_sequence_prediction": 1},
        "completion_coverage_fraction": 0.0,
        "all_requested_timestamps_completed": False,
    }
