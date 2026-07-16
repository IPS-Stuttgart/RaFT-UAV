from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.calibration.bias import SensorBiasCorrectionModel


def _model() -> SensorBiasCorrectionModel:
    return SensorBiasCorrectionModel(
        source="rf",
        target_columns=("east_m", "north_m"),
        feature_columns=(),
        intercept=np.array([1.0, -2.0]),
        coefficients=np.empty((0, 2), dtype=float),
        feature_mean=np.empty(0, dtype=float),
        feature_scale=np.empty(0, dtype=float),
        residual_std=np.array([0.5, 0.75]),
        training_rows=4,
        ridge_alpha=0.0,
        time_gate_s=1.0,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [10.0, 20.0],
            "north_m": [30.0, 40.0],
        }
    )


def test_correct_frame_omits_raw_columns_when_requested() -> None:
    frame = _frame()

    corrected = _model().correct_frame(frame, keep_uncorrected=False)

    assert "raw_east_m" not in corrected.columns
    assert "raw_north_m" not in corrected.columns
    assert "bias_east_m" in corrected.columns
    assert "bias_north_m" in corrected.columns
    np.testing.assert_allclose(corrected["east_m"], [9.0, 19.0])
    np.testing.assert_allclose(corrected["north_m"], [32.0, 42.0])
    np.testing.assert_allclose(frame["east_m"], [10.0, 20.0])
    np.testing.assert_allclose(frame["north_m"], [30.0, 40.0])


def test_correct_frame_keeps_raw_columns_by_default() -> None:
    corrected = _model().correct_frame(_frame())

    np.testing.assert_allclose(corrected["raw_east_m"], [10.0, 20.0])
    np.testing.assert_allclose(corrected["raw_north_m"], [30.0, 40.0])
