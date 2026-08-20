from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines._robust_map_lag_validation_patch import (
    _validate_record_chronology,
)
from raft_uav.baselines.robust_map import robust_map_smooth_records
from raft_uav.baselines.smoothing import smooth_tracking_records


def _record(time_s: float) -> dict[str, object]:
    return {
        "time_s": time_s,
        "source": "radar",
        "state": np.zeros(6),
        "covariance": np.eye(6),
        "accepted": True,
        "measurement_dim": 3,
    }


def test_direct_robust_map_rejects_out_of_order_records() -> None:
    with pytest.raises(ValueError, match="nondecreasing time_s"):
        robust_map_smooth_records(
            [_record(1.0), _record(0.0)],
            measurements=None,
            acceleration_std_mps2=1.0,
        )


@pytest.mark.parametrize(
    ("method", "lag_s"),
    [("robust-map", None), ("fixed-lag-map", 1.0)],
)
def test_generic_robust_map_rejects_out_of_order_records(
    method: str,
    lag_s: float | None,
) -> None:
    with pytest.raises(ValueError, match="nondecreasing time_s"):
        smooth_tracking_records(
            [_record(1.0), _record(0.0)],
            method=method,
            acceleration_std_mps2=1.0,
            lag_s=lag_s,
        )


def test_robust_map_chronology_allows_equal_timestamp_updates() -> None:
    _validate_record_chronology([_record(0.0), _record(0.0), _record(1.0)])
