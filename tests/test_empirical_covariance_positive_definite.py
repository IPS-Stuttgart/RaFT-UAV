from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.calibration.empirical_covariance import empirical_covariance_matrix
from raft_uav.uncertainty import covariance_from_row


def test_empirical_covariance_floors_degenerate_principal_directions() -> None:
    residuals = np.asarray(
        [
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    variance_floor = 0.25

    covariance = empirical_covariance_matrix(
        residuals,
        min_variance_m2=variance_floor,
    )

    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues[0] >= variance_floor - 1.0e-12

    fallback = np.diag([99.0, 101.0])
    row = pd.Series(
        {
            "cov_ee": covariance[0, 0],
            "cov_nn": covariance[1, 1],
            "cov_en": covariance[0, 1],
        }
    )
    resolved = covariance_from_row(row, 2, fallback)

    np.testing.assert_allclose(resolved, covariance)
    assert not np.array_equal(resolved, fallback)
