from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_rts_ensemble import build_track5_rts_ensemble


def _estimate(x_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [x_m],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )


def test_rts_ensemble_keeps_large_finite_weights_numerically_stable() -> None:
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )

    estimates, diagnostics = build_track5_rts_ensemble(
        [
            ("a", _estimate(0.0), 9.0e307),
            ("b", _estimate(200.0), 1.0e308),
        ],
        template,
    )

    expected_x_m = 200.0 / 1.9
    estimate = estimates.iloc[0]
    diagnostic = diagnostics.iloc[0]
    assert np.isfinite(
        estimate[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(dtype=float)
    ).all()
    assert estimate["state_x_m"] == pytest.approx(expected_x_m)
    assert diagnostic["weighted_x_m"] == pytest.approx(expected_x_m)
    assert np.isfinite(float(diagnostic["input_spread_m"]))
    assert diagnostic["inverse_variance_weight_sum"] == pytest.approx(1.9e306)
