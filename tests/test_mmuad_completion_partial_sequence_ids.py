from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["default", "seq_a"],
            "timestamp": [0.0, 0.0],
            "x": [10.0, 20.0],
            "y": [11.0, 21.0],
            "z": [12.0, 22.0],
            "uav_type": ["2", "2"],
            "score": [1.0, 1.0],
        }
    )


def test_completion_rejects_partially_missing_result_sequence_ids() -> None:
    results = _results()
    results.loc[0, "sequence_id"] = None
    template = pd.DataFrame({"sequence_id": ["seq_a"], "time_s": [0.0]})

    with pytest.raises(
        ValueError,
        match="completion results sequence IDs are partially missing",
    ):
        complete_results_to_truth_timestamps(results, template)


def test_completion_rejects_partially_missing_sequence_ids() -> None:
    template = pd.DataFrame(
        {
            "sequence_id": ["seq_a", None],
            "time_s": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="sequence IDs are partially missing"):
        complete_results_to_truth_timestamps(_results(), template)


def test_completion_rejects_partially_missing_sequence_aliases() -> None:
    template = pd.DataFrame(
        {
            "sequence": ["seq_a", " NaT "],
            "time_s": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="sequence IDs are partially missing"):
        complete_results_to_truth_timestamps(_results(), template)


def test_completion_preserves_explicit_default_sequence_name() -> None:
    template = pd.DataFrame(
        {
            "sequence_id": ["default", "seq_a"],
            "time_s": [0.0, 0.0],
        }
    )

    completed = complete_results_to_truth_timestamps(
        _results(),
        template,
        extrapolation="nan",
    )

    assert completed.rows["sequence_id"].tolist() == ["default", "seq_a"]
    assert completed.rows["x"].tolist() == [10.0, 20.0]
