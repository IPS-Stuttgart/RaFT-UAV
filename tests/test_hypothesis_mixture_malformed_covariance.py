from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.hypothesis_mixture import position_mixture_from_association_rows


@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param("not-a-number", id="text"),
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(False), id="numpy-bool"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param([1.0], id="non-scalar"),
    ],
)
def test_position_mixture_falls_back_for_malformed_association_covariance(
    invalid_value: object,
) -> None:
    rows = pd.DataFrame(
        {
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "association_score": [0.0],
            "association_cov_ee": [1.0],
            "association_cov_nn": [1.0],
            "association_cov_uu": [1.0],
            "association_cov_en": [0.0],
            "association_cov_eu": [0.0],
            "association_cov_nu": [0.0],
        }
    )
    rows["association_cov_ee"] = pd.Series([invalid_value], dtype=object)

    mean, covariance, weights = position_mixture_from_association_rows(rows)

    np.testing.assert_allclose(mean, np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(
        covariance,
        np.diag([25.0**2, 25.0**2, 35.0**2]),
    )
    np.testing.assert_allclose(weights, np.array([1.0]))
