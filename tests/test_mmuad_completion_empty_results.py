from __future__ import annotations

import numpy as np
import pandas as pd

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
