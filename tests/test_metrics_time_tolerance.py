import numpy as np
import pytest

from raft_uav.evaluation.metrics import (
    empirical_position_covariance_at_times,
    interpolate_positions_at_times,
    position_errors_at_estimates_m,
    position_errors_at_times_m,
    position_errors_m,
    sampled_position_errors_m,
)


def _sample_trajectory() -> tuple[np.ndarray, np.ndarray]:
    times = np.array([0.0, 1.0])
    positions = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times)]
    )
    return times, positions


@pytest.mark.parametrize(
    "invalid_tolerance",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(-np.inf, id="negative-infinity"),
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(False), id="numpy-bool"),
        pytest.param(np.array([1.0]), id="non-scalar-array"),
    ],
)
def test_time_gated_metrics_reject_invalid_tolerances(invalid_tolerance: object) -> None:
    times, positions = _sample_trajectory()
    calls = (
        lambda: position_errors_m(
            times,
            positions,
            times,
            positions,
            max_time_delta_s=invalid_tolerance,
        ),
        lambda: position_errors_at_estimates_m(
            times,
            positions,
            times,
            positions,
            max_time_delta_s=invalid_tolerance,
        ),
        lambda: sampled_position_errors_m(
            times,
            positions,
            times,
            positions,
            max_time_delta_s=invalid_tolerance,
        ),
        lambda: interpolate_positions_at_times(
            times,
            positions,
            times,
            max_time_delta_s=invalid_tolerance,
        ),
        lambda: position_errors_at_times_m(
            times,
            positions,
            times,
            positions,
            max_time_delta_s=invalid_tolerance,
        ),
        lambda: empirical_position_covariance_at_times(
            times,
            positions,
            times,
            positions,
            max_time_delta_s=invalid_tolerance,
        ),
    )

    for call in calls:
        with pytest.raises(ValueError, match="max_time_delta_s"):
            call()


def test_time_gated_metrics_accept_zero_and_numpy_scalar_tolerances() -> None:
    times, positions = _sample_trajectory()

    np.testing.assert_allclose(
        position_errors_m(
            times,
            positions,
            times,
            positions,
            max_time_delta_s=np.float64(0.0),
        ),
        np.zeros(2),
    )
    interpolated, valid = interpolate_positions_at_times(
        times,
        positions,
        times,
        max_time_delta_s=np.array(0.0),
    )
    np.testing.assert_allclose(interpolated, positions)
    np.testing.assert_array_equal(valid, np.ones(2, dtype=bool))
