from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement


def test_tracking_measurement_rejects_asymmetric_covariance() -> None:
    with pytest.raises(ValueError, match="measurement covariance must be symmetric"):
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([1.0, 2.0]),
            covariance=np.array([[4.0, 1.0], [0.0, 4.0]]),
            source="rf",
            _apply_runtime_calibration=False,
        )


def test_tracking_measurement_rejects_indefinite_covariance() -> None:
    with pytest.raises(
        ValueError,
        match="measurement covariance must be positive semidefinite",
    ):
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([1.0, 2.0]),
            covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
            source="rf",
            _apply_runtime_calibration=False,
        )


def test_tracking_measurement_accepts_symmetric_psd_covariance() -> None:
    covariance = np.array([[4.0, 2.0], [2.0, 1.0]])

    measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([1.0, 2.0]),
        covariance=covariance,
        source="rf",
        _apply_runtime_calibration=False,
    )

    np.testing.assert_allclose(measurement.covariance, covariance)
