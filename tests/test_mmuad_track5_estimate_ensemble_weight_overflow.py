from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble


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


def test_track5_estimate_ensemble_scales_large_finite_weights() -> None:
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )

    ensemble, diagnostics = build_track5_estimate_ensemble(
        [
            ("a", _estimate(0.0), 9.0e307),
            ("b", _estimate(2.0), 1.0e308),
        ],
        template,
    )

    estimate = ensemble.iloc[0]
    diagnostic = diagnostics.iloc[0]
    assert np.isfinite(
        estimate[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(dtype=float)
    ).all()
    assert estimate["state_x_m"] == pytest.approx(20.0 / 19.0)
    assert np.isfinite(float(estimate["ensemble_position_spread_m"]))
    assert np.isfinite(float(diagnostic["position_spread_m"]))
    assert [
        summary["weight"] for summary in diagnostics.attrs["input_summaries"]
    ] == pytest.approx([9.0e307, 1.0e308])
