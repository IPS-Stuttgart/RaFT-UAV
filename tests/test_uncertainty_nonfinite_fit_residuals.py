from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.uncertainty import fit_heteroscedastic_uncertainty_model


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 0.0, 0.0],
            "north_m": [0.0, 0.0, 0.0],
        }
    )
    rf = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [20.0, np.inf, 40.0],
            "north_m": [20.0, 30.0, 40.0],
        }
    )
    return truth, rf


def _training_rows_by_dimension(
    truth: pd.DataFrame,
    rf: pd.DataFrame,
) -> dict[str, int]:
    model = fit_heteroscedastic_uncertainty_model(
        rf=rf,
        radar=None,
        truth=truth,
    )
    return {head.dimension: head.training_rows for head in model.heads}


def test_fit_excludes_nonfinite_measurement_residuals() -> None:
    truth, rf = _frames()

    rows = _training_rows_by_dimension(truth, rf)

    assert rows == {"east": 2, "north": 3}


def test_fit_excludes_nonfinite_truth_residuals() -> None:
    truth, rf = _frames()
    rf["east_m"] = [20.0, 30.0, 40.0]
    truth.loc[1, "north_m"] = -np.inf

    rows = _training_rows_by_dimension(truth, rf)

    assert rows == {"east": 3, "north": 2}
