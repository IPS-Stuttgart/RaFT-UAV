from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import metrics_from_matches


def _submission(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * row_count,
            "time_s": np.arange(row_count, dtype=float),
            "x_m": np.zeros(row_count),
            "y_m": np.zeros(row_count),
            "z_m": np.zeros(row_count),
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 2.0],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


def test_metrics_from_matches_parses_serialized_boolean_flags() -> None:
    matches = pd.DataFrame(
        {
            "sequence_id": ["seq"] * 5,
            "time_s": np.arange(5, dtype=float),
            "truth_time_s": [0.0, np.nan, 2.0, np.nan, np.nan],
            "matched": [" TRUE ", "False", "1.0", "0", None],
            "error_2d_m": [1.0, 100.0, 3.0, 100.0, 100.0],
            "error_3d_m": [1.0, 100.0, 3.0, 100.0, 100.0],
        }
    )

    summary = metrics_from_matches(
        matches,
        submission=_submission(len(matches)),
        truth=_truth(),
    )

    assert summary["pooled"]["matched_count"] == 2
    assert summary["pooled"]["unmatched_prediction_count"] == 3
    assert summary["pooled"]["covered_truth_count"] == 2
    assert summary["pooled"]["mean_3d_m"] == pytest.approx(2.0)
    assert summary["sequences"]["seq"]["matched_count"] == 2
    assert summary["sequences"]["seq"]["mean_3d_m"] == pytest.approx(2.0)


def test_metrics_from_matches_rejects_ambiguous_flags() -> None:
    matches = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "truth_time_s": [0.0],
            "matched": ["maybe"],
            "error_2d_m": [1.0],
            "error_3d_m": [1.0],
        },
        index=[42],
    )

    with pytest.raises(
        ValueError,
        match=r"matched contains invalid Boolean values at rows \[42\]",
    ):
        metrics_from_matches(
            matches,
            submission=_submission(1),
            truth=_truth().iloc[:1].copy(),
        )
