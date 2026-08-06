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


def test_position_mixture_replaces_nonfinite_scores_with_worst_finite_score() -> None:
    rows = pd.DataFrame(
        {
            "east_m": np.arange(5, dtype=float),
            "north_m": np.zeros(5),
            "up_m": np.zeros(5),
            "association_score": [1.0, 2.0, np.inf, -np.inf, np.nan],
        }
    )

    mean, covariance, weights = position_mixture_from_association_rows(rows)

    assert np.isfinite(mean).all()
    assert np.isfinite(covariance).all()
    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]
    assert np.allclose(weights[1:], weights[1])
    assert np.isclose(mean[0], np.average(rows["east_m"], weights=weights))


def test_position_mixture_uses_equal_weights_when_all_scores_are_nonfinite() -> None:
    rows = pd.DataFrame(
        {
            "east_m": [0.0, 10.0, 20.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "association_score": [np.inf, -np.inf, np.nan],
        }
    )

    mean, covariance, weights = position_mixture_from_association_rows(rows)

    assert np.allclose(weights, np.full(3, 1.0 / 3.0))
    assert np.allclose(mean, [10.0, 0.0, 0.0])
    assert np.isfinite(covariance).all()
