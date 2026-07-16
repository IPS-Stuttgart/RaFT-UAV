import numpy as np

from raft_uav.evaluation.metrics import position_errors_m


def test_position_errors_keep_tolerance_equivalent_bracket_endpoints():
    estimate_times = np.array([0.0, 1.0])
    estimate_positions = np.column_stack(
        [estimate_times, np.zeros_like(estimate_times), np.zeros_like(estimate_times)]
    )
    truth_times = np.array([5.0e-10, 1.0 - 5.0e-10])
    truth_positions = np.column_stack(
        [truth_times, np.zeros_like(truth_times), np.zeros_like(truth_times)]
    )

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=0.0,
        dimensions=3,
    )

    assert errors.shape == (2,)
    np.testing.assert_allclose(errors, np.zeros(2), atol=1.0e-15)


def test_position_errors_keep_tolerance_equivalent_support_endpoints():
    estimate_times = np.array([0.0, 1.0])
    estimate_positions = np.column_stack(
        [estimate_times, np.zeros_like(estimate_times), np.zeros_like(estimate_times)]
    )
    truth_times = np.array([-5.0e-10, 1.0 + 5.0e-10])
    truth_x = np.array([0.0, 1.0])
    truth_positions = np.column_stack(
        [truth_x, np.zeros_like(truth_x), np.zeros_like(truth_x)]
    )

    errors = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=0.0,
        dimensions=3,
    )

    assert errors.shape == (2,)
    np.testing.assert_allclose(errors, np.zeros(2), atol=1.0e-15)
