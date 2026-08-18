import numpy as np
import pandas as pd
import pytest

from raft_uav.uncertainty import covariance_from_row


@pytest.mark.parametrize(
    "fallback",
    [
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, np.nan]]),
        np.array([[1.0, 0.1], [0.2, 1.0]]),
        np.eye(3),
        np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0]]),
        np.eye(2, dtype=bool),
        np.ma.array(
            [[1.0, 0.0], [0.0, 1.0]],
            mask=[[False, False], [False, True]],
        ),
    ],
    ids=[
        "indefinite",
        "singular",
        "nonfinite",
        "asymmetric",
        "wrong-shape",
        "complex",
        "boolean",
        "masked",
    ],
)
def test_covariance_from_row_rejects_unsafe_fallback_covariance(fallback):
    with pytest.raises(
        ValueError,
        match=r"fallback must be a finite symmetric positive-definite 2x2 covariance matrix",
    ):
        covariance_from_row(pd.Series(dtype=float), 2, fallback)


def test_covariance_from_row_keeps_valid_fallback_behavior():
    fallback = np.array([[10.0, 1.0], [1.0, 20.0]])
    row = pd.Series({"cov_ee": -1.0, "cov_nn": np.nan})

    covariance = covariance_from_row(row, 2, fallback)

    assert np.allclose(covariance, fallback)


def test_covariance_from_row_still_prefers_valid_row_covariance():
    fallback = np.diag([10.0, 20.0])
    row = pd.Series({"cov_ee": 4.0, "cov_nn": 9.0, "cov_en": 1.0})

    covariance = covariance_from_row(row, 2, fallback)

    assert np.allclose(covariance, [[4.0, 1.0], [1.0, 9.0]])


def test_covariance_from_row_preserves_dim_validation():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        covariance_from_row(pd.Series(dtype=float), 4, np.eye(4))
