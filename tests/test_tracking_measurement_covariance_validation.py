from __future__ import annotations

import numpy as np
import pytest

import raft_uav.baselines.kalman as kalman
from raft_uav.baselines.kalman import TrackingMeasurement


def _measurement(covariance: np.ndarray, *, calibrate: bool = False) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=0.0,
        vector=np.array([1.0, 2.0]),
        covariance=covariance,
        source="rf",
        _apply_runtime_calibration=calibrate,
    )


def test_tracking_measurement_rejects_asymmetric_covariance() -> None:
    with pytest.raises(ValueError, match="measurement covariance must be symmetric"):
        _measurement(np.array([[4.0, 1.0], [0.0, 4.0]]))


def test_tracking_measurement_rejects_indefinite_covariance() -> None:
    with pytest.raises(
        ValueError,
        match="measurement covariance must be positive semidefinite",
    ):
        _measurement(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_tracking_measurement_accepts_positive_semidefinite_covariance() -> None:
    covariance = np.array([[4.0, 2.0], [2.0, 1.0]])

    measurement = _measurement(covariance)

    np.testing.assert_allclose(measurement.covariance, covariance)


def test_tracking_measurement_removes_negligible_antisymmetric_roundoff() -> None:
    covariance = np.array([[4.0, 1.0 + 1.0e-13], [1.0, 2.0]])

    measurement = _measurement(covariance)

    np.testing.assert_allclose(measurement.covariance, measurement.covariance.T)
    np.testing.assert_allclose(
        measurement.covariance,
        0.5 * (covariance + covariance.T),
    )


def test_tracking_measurement_validates_calibrated_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kalman,
        "scale_covariance_for_calibrated_source",
        lambda source, dimension, covariance: np.array([[1.0, 2.0], [2.0, 1.0]]),
    )

    with pytest.raises(
        ValueError,
        match="measurement covariance must be positive semidefinite",
    ):
        _measurement(np.eye(2), calibrate=True)
