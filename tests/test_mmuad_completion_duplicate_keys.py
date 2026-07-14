from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps


def _result_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "timestamp": [2.0, 2.0],
            "x": [1.0, 100.0],
            "y": [2.0, 200.0],
            "z": [3.0, 300.0],
            "uav_type": ["2", "2"],
            "score": [0.75, 0.25],
        }
    )


def test_completion_rejects_duplicate_result_timestamps() -> None:
    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": [2.0]})

    with pytest.raises(
        ValueError,
        match=r"duplicate \(sequence_id, timestamp\) key.*seq1@2",
    ):
        complete_results_to_truth_timestamps(_result_rows(), template)


def test_completion_still_accepts_unique_result_timestamps() -> None:
    results = _result_rows().iloc[[0]].copy()
    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": [2.0]})

    completed = complete_results_to_truth_timestamps(
        results,
        template,
        extrapolation="nan",
    )

    assert completed.rows[["x", "y", "z"]].iloc[0].tolist() == [1.0, 2.0, 3.0]
    assert completed.diagnostics["completion_method"].tolist() == ["exact"]
