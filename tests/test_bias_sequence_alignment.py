from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.calibration.bias import RF_TARGET_COLUMNS
from raft_uav.calibration.bias import fit_sensor_bias_correction
from raft_uav.calibration.bias import make_bias_training_examples


def _pooled_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    measurements = pd.DataFrame(
        {
            "sequence_id": [" seqB ", "seqA", "seqC"],
            "time_s": [0.0, 0.0, 0.0],
            "east_m": [100.0, 0.0, 50.0],
            "north_m": [10.0, 0.0, 5.0],
            "std_m": [20.0, 20.0, 20.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 10.0],
        }
    )
    return measurements, truth


def test_bias_examples_do_not_cross_sequence_boundaries() -> None:
    measurements, truth = _pooled_frames()

    examples = make_bias_training_examples(
        measurements,
        truth,
        source="rf",
        target_columns=RF_TARGET_COLUMNS,
        time_gate_s=0.1,
    )

    assert examples["sequence_id"].tolist() == [" seqB ", "seqA"]
    assert examples["bias_east_m"].tolist() == [0.0, 0.0]
    assert examples["bias_north_m"].tolist() == [0.0, 0.0]


def test_bias_model_fit_uses_only_same_sequence_truth_rows() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "east_m": [0.0, 1.0, 100.0, 101.0],
            "north_m": [0.0, 2.0, 10.0, 12.0],
        }
    )
    measurements = truth.copy()
    measurements["std_m"] = 20.0

    model = fit_sensor_bias_correction(
        measurements,
        truth,
        source="rf",
        target_columns=RF_TARGET_COLUMNS,
        feature_columns=(),
        time_gate_s=0.1,
        ridge_alpha=0.0,
        min_samples=2,
    )

    assert model.training_rows == len(measurements)
    np.testing.assert_allclose(model.intercept, np.zeros(2), atol=1e-12)
    corrected = model.apply(measurements)
    np.testing.assert_allclose(
        corrected[["east_m", "north_m"]],
        measurements[["east_m", "north_m"]],
        atol=1e-12,
    )
