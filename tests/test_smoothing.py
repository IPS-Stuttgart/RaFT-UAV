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


def test_rts_smoothing_preserves_last_filtered_state_and_marks_records():
    records = run_async_cv_baseline(
        [_measurement(0.0, 0.0), _measurement(1.0, 1.0), _measurement(2.0, 2.0)]
    )

    smoothed = smooth_tracking_records(
        records,
        method="rts",
        acceleration_std_mps2=4.0,
    )

    np.testing.assert_allclose(smoothed[-1]["state"], records[-1]["state"])
    assert smoothed[0]["smoother_method"] == "rts"
    assert "filtered_state" in smoothed[0]


def test_fixed_lag_zero_returns_filtered_state():
    records = run_async_cv_baseline(
        [_measurement(0.0, 0.0), _measurement(1.0, 1.0), _measurement(2.0, 2.0)]
    )

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


@pytest.mark.parametrize("lag_s", [float("nan"), float("inf")])
def test_fixed_lag_rejects_nonfinite_lag(lag_s: float):
    records = run_async_cv_baseline(
        [_measurement(0.0, 0.0), _measurement(1.0, 1.0), _measurement(2.0, 2.0)]
    )

    with pytest.raises(ValueError, match="lag_s must be a finite nonnegative real scalar or None"):
        smooth_tracking_records(
            records,
            method="fixed-lag",
            acceleration_std_mps2=4.0,
            lag_s=lag_s,
        )


@pytest.mark.parametrize(
    "acceleration_std_mps2",
    [-1.0, float("nan"), float("inf")],
)
def test_smoothing_rejects_invalid_acceleration_std(acceleration_std_mps2: float):
    records = run_async_cv_baseline(
        [_measurement(0.0, 0.0), _measurement(1.0, 1.0), _measurement(2.0, 2.0)]
    )

    with pytest.raises(
        ValueError,
        match="acceleration_std_mps2 must be a finite nonnegative real scalar",
    ):
        smooth_tracking_records(
            records,
            method="rts",
            acceleration_std_mps2=acceleration_std_mps2,
        )
