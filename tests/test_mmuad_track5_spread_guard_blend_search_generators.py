from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_spread_guard_blend_search import (
    search_track5_spread_guard_blend_settings,
)


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [1, 1],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "x_m": [1.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


def _estimate(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [offset, 1.0 + offset],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )


def test_spread_guard_blend_search_reuses_generator_controls(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted.csv"
    outlier = tmp_path / "outlier.csv"
    _estimate(0.0).to_csv(trusted, index=False)
    _estimate(4.0).to_csv(outlier, index=False)

    grid, _ = search_track5_spread_guard_blend_settings(
        [
            EstimateInput("trusted", trusted, 0.5),
            EstimateInput("outlier", outlier, 0.5),
        ],
        template=_template(),
        truth=_truth(),
        spread_thresholds_m=(0.0, 10.0),
        fallback_blends=(value for value in (0.0, 0.5)),
        fallback_policies=(value for value in ("max-weight", "label")),
        fallback_labels=(value for value in ("trusted",)),
    )

    assert len(grid) == 8
    assert set(grid["spread_threshold_m"]) == {0.0, 10.0}
    assert set(grid["fallback_policy"]) == {"max-weight", "label"}
    assert set(grid["fallback_blend"]) == {0.0, 0.5}
