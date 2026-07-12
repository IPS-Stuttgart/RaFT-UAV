from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.hypothesis_mixture import position_mixture_from_association_rows


def test_position_mixture_falls_back_for_non_psd_association_covariance() -> None:
    rows = pd.DataFrame(
        {
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "association_score": [0.0],
            "association_cov_ee": [1.0],
            "association_cov_nn": [1.0],
            "association_cov_uu": [1.0],
            "association_cov_en": [2.0],
            "association_cov_eu": [0.0],
            "association_cov_nu": [0.0],
        }
    )

    mean, covariance, weights = position_mixture_from_association_rows(rows)

    assert np.allclose(mean, [10.0, 20.0, 30.0])
    assert np.allclose(covariance, np.diag([25.0**2, 25.0**2, 35.0**2]))
    assert np.allclose(weights, [1.0])
