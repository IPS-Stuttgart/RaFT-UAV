from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.factor_graph import (
    LeastSquaresSmoothingConfig,
    _row_position_std,
)


@pytest.fixture
def config() -> LeastSquaresSmoothingConfig:
    return LeastSquaresSmoothingConfig(measurement_std_m=25.0, rf_std_m=50.0)


def test_factor_graph_falls_back_for_malformed_standard_deviations(
    config: LeastSquaresSmoothingConfig,
) -> None:
    row = pd.Series(
        {
            "source": "radar",
            "std_east_m": "not-a-number",
            "std_north_m": 2.0,
            "std_up_m": 3.0,
        }
    )

    np.testing.assert_allclose(_row_position_std(row, config), [25.0, 25.0, 25.0])


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("rf", id="lowercase"),
        pytest.param("RF", id="uppercase"),
        pytest.param(" rf ", id="padded"),
        pytest.param(np.str_("Rf"), id="numpy-string"),
    ],
)
def test_factor_graph_normalizes_rf_source_labels(
    config: LeastSquaresSmoothingConfig,
    source: object,
) -> None:
    row = pd.Series({"source": source})

    np.testing.assert_allclose(_row_position_std(row, config), [50.0, 50.0, 50.0])


@pytest.mark.parametrize(
    "invalid_variance",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(np.inf, id="infinite"),
        pytest.param(True, id="boolean"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(np.array([1.0]), id="non-scalar"),
        pytest.param(np.ma.masked, id="masked"),
        pytest.param("not-a-number", id="text"),
    ],
)
def test_factor_graph_falls_back_for_invalid_covariance_diagonals(
    config: LeastSquaresSmoothingConfig,
    invalid_variance: object,
) -> None:
    row = pd.Series(
        {
            "source": "rf",
            "cov_ee": invalid_variance,
            "cov_nn": 4.0,
            "cov_uu": 9.0,
        }
    )

    np.testing.assert_allclose(_row_position_std(row, config), [50.0, 50.0, 50.0])


def test_factor_graph_preserves_valid_scalar_like_covariance(
    config: LeastSquaresSmoothingConfig,
) -> None:
    row = pd.Series(
        {
            "source": "radar",
            "cov_ee": np.asarray(0.0),
            "cov_nn": "4.0",
            "cov_uu": np.float64(9.0),
        }
    )

    np.testing.assert_allclose(
        _row_position_std(row, config),
        [np.sqrt(1.0e-9), 2.0, 3.0],
    )
