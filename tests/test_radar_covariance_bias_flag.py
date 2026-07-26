import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.radar_covariance import (
    BIAS_RESIDUAL_INCLUDED_COLUMN,
    RADAR_COVARIANCE_COLUMNS,
    row_radar_covariance,
)
from raft_uav.calibration.bias import BIAS_RESIDUAL_STD_COLUMN_PREFIX


def _covariance_row(flag: object) -> pd.Series:
    values = dict(
        zip(
            RADAR_COVARIANCE_COLUMNS,
            [100.0, 100.0, 100.0, 0.0, 0.0, 0.0],
        )
    )
    values.update(
        {
            f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}east_m": 3.0,
            f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}north_m": 4.0,
            f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}up_m": 5.0,
            BIAS_RESIDUAL_INCLUDED_COLUMN: flag,
        }
    )
    return pd.Series(values)


@pytest.mark.parametrize(
    "flag",
    [True, np.bool_(True), 1, np.int64(1), 1.0, np.float64(1.0), "1", "1.0", "true", "YES", "on"],
)
def test_canonical_true_bias_flags_prevent_double_inflation(flag):
    covariance = row_radar_covariance(_covariance_row(flag))

    assert covariance is not None
    np.testing.assert_allclose(np.diag(covariance), [100.0, 100.0, 100.0])


@pytest.mark.parametrize(
    "flag",
    [
        False,
        0,
        2,
        -1,
        0.5,
        1.5,
        np.inf,
        -np.inf,
        np.nan,
        "0",
        "false",
        "no",
        "off",
        "2",
        "0.5",
        "maybe",
        None,
    ],
)
def test_noncanonical_bias_flags_keep_residual_inflation(flag):
    covariance = row_radar_covariance(_covariance_row(flag))

    assert covariance is not None
    np.testing.assert_allclose(np.diag(covariance), [109.0, 116.0, 125.0])
