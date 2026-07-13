import numpy as np
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement, run_async_cv_baseline
from raft_uav.baselines.smoothing import smooth_tracking_records


def _measurement(time_s: float, east_m: float) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([east_m, 0.0, 0.0]),
        covariance=np.diag([1.0, 1.0, 1.0]),
        source="radar",
    )


def _tracking_records() -> list[dict[str, object]]:
    return run_async_cv_baseline(
        [_measurement(0.0, 0.0), _measurement(1.0, 1.0), _measurement(2.0, 2.0)]
    )


def test_rts_smoothing_preserves_last_filtered_state_and_marks_records():
    records = _tracking_records()

    smoothed = smooth_tracking_records(
        records,
        method="rts",
        acceleration_std_mps2=4.0,
    )

    np.testing.assert_allclose(smoothed[-1]["state"], records[-1]["state"])
    assert smoothed[0]["smoother_method"] == "rts"
    assert "filtered_state" in smoothed[0]


def test_fixed_lag_zero_returns_filtered_state():
    records = _tracking_records()

    smoothed = smooth_tracking_records(
        records,
        method="fixed-lag",
        acceleration_std_mps2=4.0,
        lag_s=0.0,
    )

    for original, fixed_lag in zip(records, smoothed, strict=True):
        np.testing.assert_allclose(fixed_lag["state"], original["state"])
        assert fixed_lag["smoother_lag_s"] == 0.0


def test_smoothing_does_not_mutate_input_records():
    records = run_async_cv_baseline(
        [_measurement(0.0, 0.0), _measurement(1.0, 10.0), _measurement(2.0, 2.0)]
    )
    original_state = records[0]["state"].copy()

    smooth_tracking_records(records, method="rts", acceleration_std_mps2=4.0)

    np.testing.assert_allclose(records[0]["state"], original_state)
    assert "filtered_state" not in records[0]


@pytest.mark.parametrize("method", ["rts", "fixed-lag", "robust-map", "fixed-lag-map"])
@pytest.mark.parametrize("acceleration_std_mps2", [np.nan, np.inf, -np.inf, -1.0])
def test_smoothing_rejects_invalid_acceleration_standard_deviation(
    method: str,
    acceleration_std_mps2: float,
) -> None:
    kwargs = {"lag_s": 1.0} if method in ("fixed-lag", "fixed-lag-map") else {}

    with pytest.raises(
        ValueError,
        match="acceleration_std_mps2 must be finite and nonnegative",
    ):
        smooth_tracking_records(
            _tracking_records(),
            method=method,
            acceleration_std_mps2=acceleration_std_mps2,
            **kwargs,
        )


@pytest.mark.parametrize("method", ["fixed-lag", "fixed-lag-map"])
@pytest.mark.parametrize("lag_s", [np.nan, np.inf, -np.inf, -1.0])
def test_fixed_lag_smoothing_rejects_invalid_lag(method: str, lag_s: float) -> None:
    with pytest.raises(ValueError, match="lag_s must be finite and nonnegative"):
        smooth_tracking_records(
            _tracking_records(),
            method=method,
            acceleration_std_mps2=4.0,
            lag_s=lag_s,
        )
