import numpy as np
import pytest

from raft_uav.baselines.kalman import (
    TrackingMeasurement,
    constant_velocity_matrix,
    run_async_cv_baseline,
    white_acceleration_process_noise,
)


_INVALID_NONNEGATIVE_SCALARS = (
    -1.0,
    np.nan,
    np.inf,
    -np.inf,
    True,
    1.0 + 2.0j,
    np.array([1.0]),
    np.ma.masked,
)


def test_constant_velocity_matrix_places_dt_in_position_velocity_blocks():
    matrix = constant_velocity_matrix(2.5)

    assert matrix.shape == (6, 6)
    assert matrix[0, 3] == 2.5
    assert matrix[1, 4] == 2.5
    assert matrix[2, 5] == 2.5


@pytest.mark.parametrize("dt_s", _INVALID_NONNEGATIVE_SCALARS)
def test_constant_velocity_matrix_rejects_invalid_time_steps(dt_s):
    with pytest.raises(
        ValueError,
        match="dt_s must be a finite, non-negative real scalar",
    ):
        constant_velocity_matrix(dt_s)


def test_process_noise_is_symmetric():
    covariance = white_acceleration_process_noise(0.5, 4.0)

    assert covariance.shape == (6, 6)
    np.testing.assert_allclose(covariance, covariance.T)


@pytest.mark.parametrize("dt_s", _INVALID_NONNEGATIVE_SCALARS)
def test_process_noise_rejects_invalid_time_steps(dt_s):
    with pytest.raises(
        ValueError,
        match="dt_s must be a finite, non-negative real scalar",
    ):
        white_acceleration_process_noise(dt_s, 4.0)


@pytest.mark.parametrize("acceleration_std", _INVALID_NONNEGATIVE_SCALARS)
def test_process_noise_rejects_invalid_acceleration_std(acceleration_std):
    with pytest.raises(
        ValueError,
        match="acceleration_std must be a finite, non-negative real scalar",
    ):
        white_acceleration_process_noise(0.5, acceleration_std)


def test_process_noise_matches_continuous_white_acceleration_model():
    covariance = white_acceleration_process_noise(2.0, 3.0)
    expected_block = np.array(
        [
            [24.0, 18.0],
            [18.0, 18.0],
        ]
    )

    for position_index, velocity_index in ((0, 3), (1, 4), (2, 5)):
        np.testing.assert_allclose(
            covariance[np.ix_([position_index, velocity_index], [position_index, velocity_index])],
            expected_block,
        )

    assert covariance[0, 1] == 0.0
    assert covariance[3, 4] == 0.0


def test_async_cv_baseline_returns_one_record_per_measurement():
    covariance = np.diag([10.0, 10.0, 10.0])
    measurements = [
        TrackingMeasurement(0.0, np.array([0.0, 0.0, 0.0]), covariance, "radar"),
        TrackingMeasurement(1.0, np.array([1.0, 0.0, 0.0]), covariance, "radar"),
    ]

    records = run_async_cv_baseline(measurements)

    assert len(records) == 2
    assert records[-1]["source"] == "radar"
    assert np.asarray(records[-1]["state"]).shape == (6,)
