import numpy as np
import pandas as pd
import pytest

from raft_uav.uncertainty import covariance_from_row


@pytest.mark.parametrize(
    "fallback",
    [
        np.eye(3),
        np.array([[1.0, 0.0], [0.0, np.nan]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
    ],
    ids=["wrong-shape", "non-finite", "indefinite"],
)
def test_covariance_from_row_rejects_invalid_fallback_when_used(fallback):
    with pytest.raises(ValueError, match="resolved covariance must be a finite 2x2"):
        covariance_from_row(pd.Series(dtype=float), 2, fallback)


def test_covariance_from_row_does_not_validate_unused_fallback():
    row = pd.Series({"cov_ee": 4.0, "cov_nn": 9.0, "cov_en": 1.0})
    invalid_fallback = np.array([[np.nan]])

    covariance = covariance_from_row(row, 2, invalid_fallback)

    assert np.allclose(covariance, [[4.0, 1.0], [1.0, 9.0]])
