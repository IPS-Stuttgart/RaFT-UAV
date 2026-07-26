from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.measurement_models import fit_linear_radar_bias_model


def _training_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "range_m": [0.0, 1.0, 2.0, 3.0],
            "residual_east_m": [5.0, 7.0, 9.0, 11.0],
            "residual_north_m": [0.0, 0.0, 0.0, 0.0],
            "residual_up_m": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_ridge_exempts_intercept_by_name_not_position() -> None:
    examples = _training_examples()
    model = fit_linear_radar_bias_model(
        examples,
        feature_names=("range_m", "intercept"),
        ridge_lambda=100.0,
    )

    design = np.column_stack(
        [examples["range_m"].to_numpy(dtype=float), np.ones(len(examples))]
    )
    targets = examples[
        ["residual_east_m", "residual_north_m", "residual_up_m"]
    ].to_numpy(dtype=float)
    expected = np.linalg.solve(
        design.T @ design + np.diag([100.0, 0.0]),
        design.T @ targets,
    )

    assert model.feature_names == ("range_m", "intercept")
    np.testing.assert_allclose(model.coefficients, expected)


def test_ridge_regularizes_first_feature_when_intercept_is_omitted() -> None:
    examples = _training_examples()
    model = fit_linear_radar_bias_model(
        examples,
        feature_names=("range_m",),
        ridge_lambda=100.0,
    )

    design = examples[["range_m"]].to_numpy(dtype=float)
    targets = examples[
        ["residual_east_m", "residual_north_m", "residual_up_m"]
    ].to_numpy(dtype=float)
    expected = np.linalg.solve(
        design.T @ design + np.diag([100.0]),
        design.T @ targets,
    )

    np.testing.assert_allclose(model.coefficients, expected)
