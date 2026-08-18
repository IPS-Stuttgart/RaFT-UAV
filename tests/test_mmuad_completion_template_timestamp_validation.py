from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "timestamp": [0.0],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "uav_type": ["2"],
            "score": [1.0],
        }
    )


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        "not-a-time",
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        1.0 + 2.0j,
        np.ma.masked,
    ],
)
def test_completion_rejects_malformed_template_timestamps(timestamp: object) -> None:
    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": [timestamp]})

    with pytest.raises(
        ValueError,
        match="completion template timestamps must be finite real scalars",
    ):
        complete_results_to_truth_timestamps(_results(), template)


def test_completion_accepts_numeric_string_template_timestamp() -> None:
    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": ["0.0"]})

    completed = complete_results_to_truth_timestamps(_results(), template)

    assert completed.rows["timestamp"].tolist() == [0.0]
    assert completed.diagnostics["completion_method"].tolist() == ["exact"]


def test_completion_uses_valid_timestamp_alias_for_missing_canonical_value() -> None:
    template = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [None],
            "timestamp_s": [0.0],
        }
    )

    completed = complete_results_to_truth_timestamps(_results(), template)

    assert completed.rows["timestamp"].tolist() == [0.0]
    assert completed.diagnostics["completion_method"].tolist() == ["exact"]
