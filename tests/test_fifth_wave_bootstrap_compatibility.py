from __future__ import annotations

import numpy as np

from raft_uav.evaluation.fifth_wave_diagnostics import block_bootstrap_interval


def test_block_bootstrap_preserves_block_structure_after_stability_patch() -> None:
    values = np.concatenate([np.zeros(100), np.ones(100)])

    iid = block_bootstrap_interval(
        values,
        block_size=1,
        resamples=500,
        seed=1,
    )
    blocked = block_bootstrap_interval(
        values,
        block_size=50,
        resamples=500,
        seed=1,
    )

    assert blocked.upper - blocked.lower > iid.upper - iid.lower


def test_block_bootstrap_keeps_large_representable_rmse_interval_finite() -> None:
    values = np.array([8.0e307, 1.0e308])
    expected = np.sqrt(0.82) * 1.0e308

    with np.errstate(over="raise", invalid="raise"):
        interval = block_bootstrap_interval(
            values,
            metric="rmse",
            block_size=2,
            resamples=20,
            seed=1,
        )

    assert np.isclose(interval.estimate, expected, rtol=1.0e-15)
    assert np.isfinite(interval.lower)
    assert np.isfinite(interval.upper)
