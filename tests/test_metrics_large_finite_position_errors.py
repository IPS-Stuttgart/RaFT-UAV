import numpy as np
import pytest

from raft_uav.evaluation.metrics import (
    position_errors_at_estimates_m,
    position_errors_at_times_m,
    position_errors_m,
    sampled_position_errors_m,
)


_LARGE_POSITION_M = np.array([6.0e307, 8.0e307, 0.0])
_EXPECTED_ERROR_M = 1.0e308


@pytest.mark.parametrize(
    "metric",
    [
        position_errors_m,
        position_errors_at_estimates_m,
        position_errors_at_times_m,
        sampled_position_errors_m,
    ],
)
def test_position_error_metrics_keep_large_representable_norms(metric) -> None:
    times_s = np.array([0.0, 1.0])
    estimate_positions_m = np.vstack([_LARGE_POSITION_M, _LARGE_POSITION_M])
    truth_positions_m = np.zeros((2, 3), dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        errors_m = metric(
            times_s,
            estimate_positions_m,
            times_s,
            truth_positions_m,
            dimensions=3,
        )

    assert errors_m.shape == (2,)
    assert bool(np.isfinite(errors_m).all())
    np.testing.assert_allclose(
        errors_m,
        np.full(2, _EXPECTED_ERROR_M),
        rtol=1.0e-12,
        atol=0.0,
    )


def test_single_sample_truth_grid_keeps_large_representable_norm() -> None:
    with np.errstate(over="raise", invalid="raise"):
        errors_m = position_errors_m(
            np.array([0.0]),
            _LARGE_POSITION_M.reshape(1, 3),
            np.array([0.0]),
            np.zeros((1, 3), dtype=float),
            dimensions=3,
        )

    assert errors_m.shape == (1,)
    assert np.isfinite(errors_m[0])
    assert errors_m[0] == pytest.approx(_EXPECTED_ERROR_M, rel=1.0e-12)
