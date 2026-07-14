from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_risk_reservoir import attach_candidate_risk_score
from raft_uav.mmuad.candidate_risk_reservoir import build_risk_adjusted_reservoir


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["lidar_360", "livox_avia"],
            "track_id": ["low-risk", "high-risk"],
            "x_m": [0.0, 20.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "candidate_class_calibrated_score": [0.8, 0.9],
            "predicted_sigma_m": [1.0, 20.0],
        }
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "operation",
    [attach_candidate_risk_score, build_risk_adjusted_reservoir],
)
def test_risk_reservoir_rejects_nonfinite_uncertainty_weight(
    value: float,
    operation,
) -> None:
    with pytest.raises(ValueError, match="uncertainty_weight must be finite"):
        operation(_candidate_rows(), uncertainty_weight=value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "operation",
    [attach_candidate_risk_score, build_risk_adjusted_reservoir],
)
def test_risk_reservoir_rejects_nonfinite_sigma_floor(
    value: float,
    operation,
) -> None:
    with pytest.raises(ValueError, match="sigma_floor_m must be finite"):
        operation(_candidate_rows(), sigma_floor_m=value)


def test_risk_reservoir_accepts_finite_numpy_scalar_controls() -> None:
    scored = attach_candidate_risk_score(
        _candidate_rows(),
        uncertainty_weight=np.float64(0.5),
        sigma_floor_m=np.float32(2.0),
    ).rows

    assert np.isfinite(scored["candidate_risk_adjusted_score"]).all()
    assert scored["candidate_risk_uncertainty_weight"].eq(0.5).all()
    assert scored["candidate_risk_sigma_floor_m"].eq(2.0).all()
