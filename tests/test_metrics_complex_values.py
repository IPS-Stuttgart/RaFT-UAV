import numpy as np
import pytest

from raft_uav.evaluation.metrics import (
    interpolate_positions_at_times,
    nearest_time_indices,
    position_errors_at_estimates_m,
)


def _trajectory() -> tuple[np.ndarray, np.ndarray]:
    times = np.array([0.0, 1.0])
    positions = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times)]
    )
    return times, positions


def test_nearest_time_indices_rejects_complex_reference_timestamps() -> None:
    with pytest.raises(ValueError, match="reference_times_s"):
        nearest_time_indices(
            np.array([0.0 + 2.0j, 1.0 + 0.0j]),
            np.array([0.5]),
        )


def test_nearest_time_indices_rejects_complex_query_timestamps() -> None:
    with pytest.raises(ValueError, match="query_times_s"):
        nearest_time_indices(
            np.array([0.0, 1.0]),
            np.array([0.5 + 3.0j]),
        )


def test_position_errors_rejects_complex_coordinates() -> None:
    truth_times, truth_positions = _trajectory()
    estimate_positions = truth_positions.astype(complex)
    estimate_positions[1, 0] += 4.0j

    with pytest.raises(ValueError, match="positions_m"):
        position_errors_at_estimates_m(
            truth_times,
            estimate_positions,
            truth_times,
            truth_positions,
        )


def test_interpolation_rejects_complex_reference_coordinates() -> None:
    times, positions = _trajectory()
    complex_positions = positions.astype(complex)
    complex_positions[1, 1] += 5.0j

    with pytest.raises(ValueError, match="reference_positions_m"):
        interpolate_positions_at_times(
            times,
            complex_positions,
            np.array([0.5]),
        )
