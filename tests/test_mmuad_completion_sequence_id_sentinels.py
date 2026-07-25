from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps
from raft_uav.mmuad.schema import normalize_truth_columns


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["default", "default"],
            "timestamp": [0.0, 1.0],
            "x": [0.0, 1.0],
            "y": [0.0, 2.0],
            "z": [0.0, 3.0],
            "uav_type": ["2", "2"],
            "score": [1.0, 1.0],
        }
    )


def test_truth_normalization_treats_nat_as_missing_sequence_id() -> None:
    truth = normalize_truth_columns(
        pd.DataFrame(
            {
                "sequence_id": [" NaT "],
                "time_s": [0.0],
                "x_m": [1.0],
                "y_m": [2.0],
                "z_m": [3.0],
            }
        )
    )

    assert truth["sequence_id"].tolist() == ["default"]


def test_timestamp_only_template_normalizes_missing_like_sequence_ids() -> None:
    template = pd.DataFrame(
        {
            "sequence_id": [" NaN ", "<NA>", "NaT", " none ", ""],
            "time_s": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    )

    completed = complete_results_to_truth_timestamps(
        _results(),
        template,
        max_interpolation_gap_s=1.0,
        extrapolation="nan",
    )

    assert completed.rows["sequence_id"].tolist() == ["default"] * 5
    assert completed.rows["timestamp"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert completed.diagnostics["completion_method"].tolist() == [
        "exact",
        "interpolated",
        "interpolated",
        "interpolated",
        "exact",
    ]


def test_template_sequence_normalization_removes_canonical_duplicates() -> None:
    template = pd.DataFrame(
        {
            "sequence_id": ["default", "NaT"],
            "time_s": [0.0, 0.0],
        }
    )

    completed = complete_results_to_truth_timestamps(
        _results(),
        template,
        extrapolation="nan",
    )

    assert len(completed.rows) == 1
    assert len(completed.diagnostics) == 1
