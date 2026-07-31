from __future__ import annotations

import pandas as pd

from raft_uav.calibration.bias import make_bias_training_examples


def _measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [2.0, 4.0],
            "north_m": [0.0, 0.0],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [1.0, 1.0],
            "north_m": [0.0, 0.0],
        }
    )


def _examples(measurements: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    return make_bias_training_examples(
        measurements,
        truth,
        source="rf",
        target_columns=("east_m", "north_m"),
        time_gate_s=0.1,
    )


def test_bias_examples_reject_nonreal_measurement_without_losing_real_row() -> None:
    measurements = _measurements().astype({"east_m": complex})
    measurements.loc[0, "east_m"] = 2.0 + 3.0j

    rows = _examples(measurements, _truth())

    assert rows["time_s"].tolist() == [1.0]
    assert rows["bias_east_m"].tolist() == [3.0]


def test_bias_examples_reject_nonreal_truth_without_losing_real_row() -> None:
    truth = _truth().astype({"east_m": complex})
    truth.loc[0, "east_m"] = 1.0 + 2.0j

    rows = _examples(_measurements(), truth)

    assert rows["time_s"].tolist() == [1.0]
    assert rows["bias_east_m"].tolist() == [3.0]


def test_bias_examples_preserve_zero_imaginary_values() -> None:
    measurements = _measurements().astype({"east_m": complex})

    rows = _examples(measurements, _truth())

    assert rows["time_s"].tolist() == [0.0, 1.0]
    assert rows["bias_east_m"].tolist() == [1.0, 3.0]
