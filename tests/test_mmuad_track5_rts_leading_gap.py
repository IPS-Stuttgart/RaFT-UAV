from __future__ import annotations

import numpy as np

from raft_uav.mmuad.track5_rts_ensemble import _rts_smooth_axis


def test_rts_smoother_initializes_at_first_observation_after_leading_gap() -> None:
    times = np.asarray([0.0, 100.0, 101.0])
    measurements = np.asarray([np.nan, 0.0, 10.0])
    variances = np.asarray([np.inf, 1.0, 1.0])
    controls = {
        "process_accel_std_mps2": 0.0,
        "initial_position_std_m": 100.0,
        "initial_velocity_std_mps": 25.0,
    }

    smoothed = _rts_smooth_axis(
        times,
        measurements,
        variances,
        **controls,
    )
    suffix_only = _rts_smooth_axis(
        times[1:],
        measurements[1:],
        variances[1:],
        **controls,
    )

    np.testing.assert_allclose(smoothed[1:], suffix_only, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(smoothed[1:], [0.0, 10.0], rtol=0.0, atol=0.05)
    assert smoothed[0] < -900.0
