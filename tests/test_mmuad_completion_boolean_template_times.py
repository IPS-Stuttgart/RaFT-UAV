from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "timestamp": [0.0, 2.0],
            "x": [0.0, 2.0],
            "y": [0.0, 0.0],
            "z": [1.0, 1.0],
            "uav_type": ["2", "2"],
            "score": [1.0, 1.0],
        }
    )


def _nested_boolean(value: bool = True) -> np.ndarray:
    inner = np.asarray(value)
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    ("column", "timestamp"),
    [
        ("time_s", True),
        ("time_s", np.bool_(False)),
        ("timestamp", np.asarray(True)),
        ("Timestamp", _nested_boolean(False)),
    ],
)
def test_completion_rejects_boolean_template_timestamps(
    column: str,
    timestamp: object,
) -> None:
    template = pd.DataFrame({"sequence_id": ["seq1"], column: [timestamp]})

    with pytest.raises(
        ValueError,
        match="completion template time_s must not contain Boolean values",
    ):
        complete_results_to_truth_timestamps(_results(), template)


def test_completion_retains_numeric_string_template_timestamps() -> None:
    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": ["1.0"]})

    completed = complete_results_to_truth_timestamps(
        _results(),
        template,
        max_interpolation_gap_s=2.0,
    )

    assert completed.rows["timestamp"].tolist() == [1.0]
    assert completed.rows["x"].tolist() == [1.0]
    assert completed.diagnostics["completion_method"].tolist() == ["interpolated"]
