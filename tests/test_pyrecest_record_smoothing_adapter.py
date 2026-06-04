from __future__ import annotations

import numpy as np

from raft_uav.baselines.smoothing import smooth_tracking_records


def _records(state_dim: int = 6) -> list[dict[str, object]]:
    records = []
    for idx, time_s in enumerate([0.0, 1.0, 2.0]):
        state = np.zeros(state_dim)
        state[0] = float(idx) + (0.25 if idx == 1 else 0.0)
        state[3] = 1.0
        covariance = np.eye(state_dim) * (2.0 - 0.4 * idx)
        records.append(
            {
                "time_s": time_s,
                "state": state,
                "covariance": covariance,
                "source": "radar" if idx != 1 else "rf",
                "accepted": True,
            }
        )
    return records


def test_fixed_lag_smoothing_delegates_to_pyrecest_record_smoother() -> None:
    smoothed = smooth_tracking_records(
        _records(),
        method="fixed-lag",
        lag_s=5.0,
        acceleration_std_mps2=0.1,
    )

    assert len(smoothed) == 3
    assert smoothed[0]["source"] == "radar"
    assert smoothed[1]["source"] == "rf"
    assert smoothed[0]["smoother_method"] == "fixed-lag"
    assert smoothed[0]["smoother_lag_s"] == 5.0
    assert np.allclose(smoothed[0]["filtered_state"], _records()[0]["state"])
    assert smoothed[0]["state"].shape == (6,)
    assert smoothed[0]["covariance"].shape == (6, 6)


def test_rts_smoothing_preserves_extra_bias_dimensions() -> None:
    smoothed = smooth_tracking_records(
        _records(state_dim=7),
        method="rts",
        acceleration_std_mps2=0.1,
    )

    assert len(smoothed) == 3
    assert smoothed[0]["state"].shape == (7,)
    assert smoothed[0]["covariance"].shape == (7, 7)
    assert smoothed[0]["smoother_method"] == "rts"
    assert smoothed[0]["smoother_lag_s"] is None
