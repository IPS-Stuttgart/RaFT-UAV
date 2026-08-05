from __future__ import annotations

import pandas as pd

from raft_uav.calibration.bias import make_bias_training_examples


def test_bias_examples_use_final_duplicate_truth_row() -> None:
    measurements = pd.DataFrame(
        {
            "time_s": [0.0, 0.05, 1.0],
            "east_m": [12.0, 13.0, 23.0],
            "north_m": [2.0, 2.0, 3.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0],
            "east_m": [100.0, 10.0, 20.0],
            "north_m": [100.0, 2.0, 3.0],
        }
    )

    rows = make_bias_training_examples(
        measurements,
        truth,
        source="rf",
        target_columns=("east_m", "north_m"),
        time_gate_s=0.1,
    )

    assert rows["time_s"].tolist() == [0.0, 0.05, 1.0]
    assert rows["bias_east_m"].tolist() == [2.0, 3.0, 3.0]
    assert rows["bias_north_m"].tolist() == [0.0, 0.0, 0.0]


def test_bias_examples_keep_latest_valid_duplicate_truth_row() -> None:
    measurements = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [12.0],
            "north_m": [2.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 0.0],
            "east_m": [100.0, 10.0, complex(8.0, 1.0)],
            "north_m": [100.0, 2.0, 2.0],
        }
    )

    rows = make_bias_training_examples(
        measurements,
        truth,
        source="rf",
        target_columns=("east_m", "north_m"),
        time_gate_s=0.0,
    )

    assert rows["bias_east_m"].tolist() == [2.0]
    assert rows["bias_north_m"].tolist() == [0.0]
