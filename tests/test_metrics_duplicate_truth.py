import numpy as np
import pytest

from raft_uav.evaluation.metrics import (
    position_errors_at_estimates_m,
    sampled_position_errors_m,
)


@pytest.mark.parametrize(
    "metric",
    [position_errors_at_estimates_m, sampled_position_errors_m],
)
def test_nearest_truth_metrics_keep_last_duplicate_truth_sample(metric):
    truth_times = np.array([1.0, 0.0, 0.0])
    truth_positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    estimate_times = np.array([0.0, 1.0])
    estimate_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    errors = metric(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        dimensions=3,
    )

    np.testing.assert_allclose(errors, np.zeros(2))
