import numpy as np
import pandas as pd

from raft_uav.evaluation.diagnostics import _position_error_frame


def test_compact_diagnostics_keep_large_representable_errors() -> None:
    estimate = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [6.0e307, 6.0e307],
            "north_m": [8.0e307, 8.0e307],
            "up_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 0.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        errors = _position_error_frame(
            estimate_frame=estimate,
            truth=truth,
            max_eval_time_delta_s=0.0,
        )

    assert len(errors) == 2
    assert bool(np.isfinite(errors["error_2d_m"]).all())
    assert bool(np.isfinite(errors["error_3d_m"]).all())
    np.testing.assert_allclose(
        errors["error_2d_m"].to_numpy(float),
        np.full(2, 1.0e308),
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        errors["error_3d_m"].to_numpy(float),
        np.full(2, 1.0e308),
        rtol=1.0e-12,
        atol=0.0,
    )
