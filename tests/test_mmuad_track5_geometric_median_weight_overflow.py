from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_geometric_median_ensemble import (
    build_track5_geometric_median_ensemble,
    weighted_geometric_median,
)


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


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )


def test_weighted_geometric_median_scales_large_finite_weights() -> None:
    center, iterations, displacement = weighted_geometric_median(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        np.asarray([1.0e308, 1.0e308, 1.0e308]),
    )

    assert np.isfinite(center).all()
    assert center == pytest.approx([1.0, 0.0, 0.0])
    assert iterations >= 1
    assert displacement == pytest.approx(0.0)


def test_track5_geomedian_scales_large_weights_and_restores_metadata() -> None:
    estimates, diagnostics = build_track5_geometric_median_ensemble(
        [
            ("left", _estimate(0.0), 9.0e307),
            ("middle", _estimate(1.0), 1.0e308),
            ("right", _estimate(2.0), 8.0e307),
        ],
        _template(),
    )

    estimate = estimates.iloc[0]
    diagnostic = diagnostics.iloc[0]
    xyz = estimate[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(dtype=float)
    assert np.isfinite(xyz).all()
    assert estimate["state_x_m"] == pytest.approx(1.0, abs=1.0e-4)
    assert np.isfinite(float(estimate["geomedian_position_spread_m"]))
    assert np.isfinite(float(diagnostic["position_spread_m"]))
    restored_weights = [
        summary["weight"] for summary in diagnostics.attrs["input_summaries"]
    ]
    assert restored_weights == pytest.approx([9.0e307, 1.0e308, 8.0e307])
