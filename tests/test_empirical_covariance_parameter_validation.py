from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.empirical_covariance import aligned_residuals
from raft_uav.calibration.empirical_covariance import empirical_covariance_matrix
from raft_uav.calibration.empirical_covariance import (
    estimate_empirical_measurement_covariances,
)


@pytest.mark.parametrize(
    "gate",
    [
        -0.1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        1.0 + 0.0j,
        np.asarray([0.5]),
        np.ma.masked,
    ],
)
def test_empirical_covariance_rejects_invalid_time_gates(gate: object) -> None:
    with pytest.raises(ValueError, match="max_time_delta_s"):
        aligned_residuals(
            pd.DataFrame(),
            pd.DataFrame(),
            source="rf",
            max_time_delta_s=gate,
        )


@pytest.mark.parametrize(
    "floor",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        1.0 + 0.0j,
        np.asarray([1.0]),
        np.ma.masked,
    ],
)
def test_empirical_covariance_rejects_invalid_variance_floors(
    floor: object,
) -> None:
    with pytest.raises(ValueError, match="min_variance_m2"):
        empirical_covariance_matrix(
            np.asarray([[1.0, -1.0]]),
            min_variance_m2=floor,
        )


def test_empirical_covariance_accepts_zero_dimensional_scalar_controls() -> None:
    truth = pd.DataFrame(
        {"time_s": [0.0], "east_m": [0.0], "north_m": [0.0]}
    )
    rf = pd.DataFrame(
        {"time_s": [0.0], "east_m": [1.0], "north_m": [-1.0]}
    )

    payload = estimate_empirical_measurement_covariances(
        rf=rf,
        radar=None,
        truth=truth,
        max_time_delta_s=np.asarray(0.0),
        min_variance_m2=np.asarray(0.5),
    )

    assert payload["max_time_delta_s"] == 0.0
    assert payload["min_variance_m2"] == 0.5
    np.testing.assert_allclose(payload["rf"]["covariance_m2"], np.eye(2) * 0.5)
