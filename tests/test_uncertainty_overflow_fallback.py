from fractions import Fraction

import numpy as np
import pandas as pd

from raft_uav.uncertainty import covariance_from_row


def test_covariance_from_row_falls_back_for_overflowing_diagonal_value() -> None:
    fallback = np.diag([10.0, 20.0])
    row = pd.Series(
        {
            "cov_ee": Fraction(10**10000, 1),
            "cov_nn": 4.0,
        }
    )

    covariance = covariance_from_row(row, 2, fallback)

    np.testing.assert_allclose(covariance, fallback)


def test_covariance_from_row_ignores_overflowing_cross_covariance() -> None:
    fallback = np.diag([10.0, 20.0])
    row = pd.Series(
        {
            "cov_ee": 4.0,
            "cov_nn": 9.0,
            "cov_en": Fraction(10**10000, 1),
        }
    )

    covariance = covariance_from_row(row, 2, fallback)

    np.testing.assert_allclose(covariance, np.diag([4.0, 9.0]))
