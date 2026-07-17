from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement


_VECTOR = np.array([1.0, 2.0, 3.0])
_COVARIANCE = np.eye(3)


def _measurement(time_s: object) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=_VECTOR,
        covariance=_COVARIANCE,
        source="radar",
        _apply_runtime_calibration=False,
    )


@pytest.mark.parametrize(
    "time_s",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
        None,
        "invalid",
    ],
)
def test_tracking_measurement_rejects_invalid_timestamps(time_s: object) -> None:
    with pytest.raises(
        ValueError,
        match="measurement time_s must be a finite real scalar",
    ):
        _measurement(time_s)


@pytest.mark.parametrize("time_s", [1.25, "1.25", np.float64(1.25), np.array(1.25)])
def test_tracking_measurement_normalizes_valid_scalar_timestamps(time_s: object) -> None:
    measurement = _measurement(time_s)

    assert measurement.time_s == pytest.approx(1.25)
    assert isinstance(measurement.time_s, float)
