from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.robust_map import robust_map_smooth_records
from raft_uav.baselines.smoothing import smooth_tracking_records


def _record() -> dict[str, object]:
    return {
        "time_s": 0.0,
        "source": "radar",
        "state": np.zeros(6),
        "covariance": np.eye(6),
        "accepted": True,
        "measurement_dim": 3,
    }


@pytest.mark.parametrize(
    "invalid_lag_s",
    [
        np.nan,
        np.inf,
        True,
        np.bool_(False),
        np.array([1.0]),
        1.0 + 0.0j,
        np.ma.masked,
    ],
)
def test_direct_robust_map_rejects_invalid_lag_horizons(
    invalid_lag_s: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="lag_s must be a finite nonnegative real scalar",
    ):
        robust_map_smooth_records(
            [_record()],
            measurements=None,
            acceleration_std_mps2=1.0,
            lag_s=invalid_lag_s,
        )


@pytest.mark.parametrize("method", ["fixed-lag", "fixed-lag-map"])
def test_generic_smoother_rejects_nan_fixed_lag_horizon(method: str) -> None:
    with pytest.raises(
        ValueError,
        match="lag_s must be a finite nonnegative real scalar",
    ):
        smooth_tracking_records(
            [_record()],
            method=method,
            acceleration_std_mps2=1.0,
            lag_s=np.nan,
        )


def test_fixed_lag_map_normalizes_scalar_array_horizon() -> None:
    smoothed = smooth_tracking_records(
        [_record()],
        method="fixed-lag-map",
        acceleration_std_mps2=1.0,
        lag_s=np.array(0.0),
    )

    assert smoothed[0]["smoother_method"] == "fixed-lag-map"
    assert smoothed[0]["smoother_lag_s"] == 0.0
