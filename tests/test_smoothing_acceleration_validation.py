import numpy as np
import pytest

from raft_uav.baselines.robust_map import robust_map_smooth_records
from raft_uav.baselines.smoothing import smooth_tracking_records


def _record() -> dict[str, object]:
    return {
        "time_s": 0.0,
        "state": np.zeros(6),
        "covariance": np.eye(6),
    }


_INVALID_ACCELERATION_STDS = (
    -1.0,
    np.nan,
    np.inf,
    -np.inf,
    True,
    np.bool_(False),
    1.0 + 0.0j,
    np.ma.masked,
    np.array([1.0]),
)


@pytest.mark.parametrize(
    ("method", "lag_s"),
    (
        ("rts", None),
        ("fixed-lag", 0.0),
        ("robust-map", None),
        ("fixed-lag-map", 0.0),
    ),
)
@pytest.mark.parametrize("acceleration_std_mps2", _INVALID_ACCELERATION_STDS)
def test_active_smoothing_rejects_invalid_acceleration_noise(
    method: str,
    lag_s: float | None,
    acceleration_std_mps2: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="acceleration_std_mps2 must be a finite nonnegative real scalar",
    ):
        smooth_tracking_records(
            [_record()],
            method=method,
            acceleration_std_mps2=acceleration_std_mps2,
            lag_s=lag_s,
        )


@pytest.mark.parametrize("acceleration_std_mps2", _INVALID_ACCELERATION_STDS)
def test_direct_robust_map_rejects_invalid_acceleration_noise(
    acceleration_std_mps2: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="acceleration_std_mps2 must be a finite nonnegative real scalar",
    ):
        robust_map_smooth_records(
            [_record()],
            measurements=None,
            acceleration_std_mps2=acceleration_std_mps2,
        )


def test_smoothing_noop_preserves_invalid_acceleration_fast_paths() -> None:
    records = [_record()]

    copied = smooth_tracking_records(
        records,
        method="none",
        acceleration_std_mps2=np.nan,
    )
    assert copied is not records
    np.testing.assert_allclose(copied[0]["state"], records[0]["state"])
    assert smooth_tracking_records(
        [],
        method="rts",
        acceleration_std_mps2=np.nan,
    ) == []
    assert robust_map_smooth_records(
        [],
        measurements=None,
        acceleration_std_mps2=np.nan,
    ) == []
