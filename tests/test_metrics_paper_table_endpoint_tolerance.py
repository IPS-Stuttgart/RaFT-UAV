import numpy as np

from raft_uav.evaluation.metrics import (
    interpolate_positions_at_times,
    position_errors_at_times_m,
)


def test_paper_table_interpolation_accepts_tolerance_equivalent_endpoints():
    reference_times = np.array([0.0, 1.0])
    reference_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    endpoint_offset_s = 5.0e-10
    query_times = np.array(
        [
            reference_times[0] - endpoint_offset_s,
            reference_times[-1] + endpoint_offset_s,
        ]
    )

    interpolated, valid = interpolate_positions_at_times(
        reference_times,
        reference_positions,
        query_times,
        max_time_delta_s=0.0,
    )
    errors = position_errors_at_times_m(
        query_times,
        reference_positions,
        reference_times,
        reference_positions,
        max_time_delta_s=0.0,
        dimensions=3,
    )

    np.testing.assert_array_equal(valid, np.array([True, True]))
    np.testing.assert_allclose(interpolated, reference_positions)
    np.testing.assert_allclose(errors, np.zeros(2))
