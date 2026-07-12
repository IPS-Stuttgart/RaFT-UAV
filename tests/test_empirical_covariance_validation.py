from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.empirical_covariance import (
    aligned_residuals,
    empirical_covariance_matrix,
    estimate_empirical_measurement_covariances,
)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_empirical_covariance_rejects_invalid_variance_floor(value: float) -> None:
    residuals = np.asarray([[1.0, -1.0]], dtype=float)

    with pytest.raises(ValueError, match="min_variance_m2 must be positive and finite"):
        empirical_covariance_matrix(residuals, min_variance_m2=value)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_aligned_residuals_rejects_invalid_time_gate(value: float) -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [1.0],
            "north_m": [1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
        }
    )

    with pytest.raises(ValueError, match="max_time_delta_s must be finite and non-negative"):
        aligned_residuals(
            frame,
            truth,
            source="rf",
            max_time_delta_s=value,
        )


def test_estimator_validates_controls_before_skipping_empty_sources() -> None:
    with pytest.raises(ValueError, match="min_variance_m2 must be positive and finite"):
        estimate_empirical_measurement_covariances(
            rf=None,
            radar=None,
            truth=pd.DataFrame(),
            min_variance_m2=-1.0,
        )
