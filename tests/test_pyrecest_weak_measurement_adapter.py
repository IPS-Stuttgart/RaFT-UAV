from __future__ import annotations

import numpy as np

from pyrecest.models import (
    MaskedLinearMeasurementModel,
    WeakDimensionMeasurementModel,
    diagonal_measurement_covariance,
    selection_matrix,
)

from raft_uav.baselines.kalman import TrackingMeasurement, measurement_matrix


def test_cv_measurement_matrix_uses_pyrecest_selection_convention() -> None:
    assert np.allclose(measurement_matrix(2), selection_matrix(6, [0, 1]))
    assert np.allclose(measurement_matrix(3), selection_matrix(6, [0, 1, 2]))


def test_tracking_measurement_accepts_pyrecest_weak_dimension_covariance() -> None:
    covariance = diagonal_measurement_covariance([360.0, 360.0, 20000.0])

    measurement = TrackingMeasurement(
        time_s=1.0,
        vector=np.array([1.0, 2.0, 3.0]),
        covariance=covariance,
        source="radar",
    )

    assert np.isclose(measurement.covariance[0, 0], 360.0**2)
    assert np.isclose(measurement.covariance[2, 2], 20000.0**2)


def test_pyrecest_masked_and_weak_models_expose_linear_update_attributes() -> None:
    masked = MaskedLinearMeasurementModel(state_dim=6, observed_dims=[0, 1], stds=[75.0, 75.0])
    weak = WeakDimensionMeasurementModel(
        measurement_matrix(3),
        stds=[360.0, 360.0, 20000.0],
    )

    assert masked.measurement_matrix.shape == (2, 6)
    assert weak.measurement_matrix.shape == (3, 6)
    assert np.isclose(weak.measurement_noise_cov[2, 2], 20000.0**2)
