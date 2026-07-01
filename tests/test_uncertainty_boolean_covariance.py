import numpy as np
import pandas as pd

from raft_uav.uncertainty import covariance_from_row


def test_covariance_from_row_falls_back_for_boolean_covariance_values():
    fallback = np.diag([10.0, 20.0])
    row = pd.Series({"cov_ee": True, "cov_nn": np.bool_(True)})

    cov = covariance_from_row(row, 2, fallback)

    assert np.allclose(cov, fallback)
