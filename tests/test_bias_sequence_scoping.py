from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.calibration.bias import make_bias_training_examples


def test_bias_examples_match_truth_within_each_sequence() -> None:
    measurements = pd.DataFrame(
        {
            "sequence_id": ["flight-a", "flight-b", "flight-a", "flight-b"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "east_m": [11.0, 101.0, 21.0, 111.0],
            "north_m": [1.0, 10.0, 2.0, 11.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["flight-a", "flight-b", "flight-a", "flight-b"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "east_m": [10.0, 100.0, 20.0, 110.0],
            "north_m": [1.0, 10.0, 2.0, 11.0],
        }
    )

    rows = make_bias_training_examples(
        measurements,
        truth,
        source="rf",
        target_columns=("east_m", "north_m"),
        time_gate_s=0.0,
    )

    assert rows["sequence_id"].tolist() == [
        "flight-a",
        "flight-b",
        "flight-a",
        "flight-b",
    ]
    assert rows["bias_east_m"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert rows["bias_north_m"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_bias_examples_reject_ambiguous_one_sided_sequence_labels() -> None:
    measurements = pd.DataFrame(
        {
            "sequence_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "east_m": [11.0, 101.0],
            "north_m": [1.0, 10.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [10.0, 20.0],
            "north_m": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="on both measurements and truth"):
        make_bias_training_examples(
            measurements,
            truth,
            source="rf",
            target_columns=("east_m", "north_m"),
            time_gate_s=0.1,
        )
