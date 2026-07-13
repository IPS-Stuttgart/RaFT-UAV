from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.uncertainty import _aligned_residuals
from raft_uav.uncertainty import fit_heteroscedastic_uncertainty_model


def _pooled_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    measurements = pd.DataFrame(
        {
            "sequence_id": ["seqB", "seqA", "seqC"],
            "time_s": [0.0, 0.0, 0.0],
            "east_m": [100.0, 0.0, 50.0],
            "north_m": [10.0, 0.0, 5.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 10.0],
            "up_m": [0.0, 0.0],
        }
    )
    return measurements, truth


def test_uncertainty_residuals_do_not_cross_sequence_boundaries() -> None:
    measurements, truth = _pooled_frames()

    aligned = _aligned_residuals(
        measurements,
        truth,
        max_time_delta_s=0.1,
    )

    assert aligned["sequence_id"].tolist() == ["seqB", "seqA"]
    assert aligned["residual_east_m"].tolist() == [0.0, 0.0]
    assert aligned["residual_north_m"].tolist() == [0.0, 0.0]


def test_uncertainty_fit_uses_only_same_sequence_truth_rows() -> None:
    measurements, truth = _pooled_frames()

    model = fit_heteroscedastic_uncertainty_model(
        rf=measurements,
        radar=None,
        truth=truth,
        ridge_lambda=1.0,
        max_time_delta_s=0.1,
        min_std_m={"rf": {"east": 0.1, "north": 0.1}},
        max_std_m={"rf": {"east": 100.0, "north": 100.0}},
    )

    rf_heads = {head.dimension: head for head in model.heads if head.source == "rf"}
    assert rf_heads["east"].training_rows == 2
    assert rf_heads["north"].training_rows == 2
    assert np.exp(rf_heads["east"].coefficients[0]) == pytest.approx(0.01)
    assert np.exp(rf_heads["north"].coefficients[0]) == pytest.approx(0.01)
