from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.uncertainty import fit_heteroscedastic_uncertainty_model


def _training_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    rf = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [1.0, 2.0],
            "north_m": [1.0, 1.0],
        }
    )
    return rf, truth


@pytest.mark.parametrize(
    "ridge_lambda",
    [-1.0, np.nan, np.inf, True, np.array([1.0])],
)
def test_uncertainty_fit_rejects_invalid_ridge_lambda(ridge_lambda: object) -> None:
    rf, truth = _training_frames()

    with pytest.raises(ValueError, match="ridge_lambda"):
        fit_heteroscedastic_uncertainty_model(
            rf=rf,
            radar=None,
            truth=truth,
            ridge_lambda=ridge_lambda,
        )


@pytest.mark.parametrize(
    "max_time_delta_s",
    [-0.1, np.nan, np.inf, False, np.array([1.0])],
)
def test_uncertainty_fit_rejects_invalid_time_gate(max_time_delta_s: object) -> None:
    rf, truth = _training_frames()

    with pytest.raises(ValueError, match="max_time_delta_s"):
        fit_heteroscedastic_uncertainty_model(
            rf=rf,
            radar=None,
            truth=truth,
            max_time_delta_s=max_time_delta_s,
        )
