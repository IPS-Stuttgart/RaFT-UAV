from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_spread_guard_blend_search import (
    search_track5_spread_guard_blend_settings,
    write_spread_guard_blend_search_outputs,
)


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2],
        }
    )


def _estimate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )


def _unmatched_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["different", "different"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


def test_blend_search_rejects_grid_without_finite_truth_matches(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimate_rows().to_csv(estimate_csv, index=False)

    with pytest.raises(
        ValueError,
        match="no spread-guard blend candidate had finite matched truth rows",
    ):
        search_track5_spread_guard_blend_settings(
            [EstimateInput("estimate", estimate_csv)],
            template=_template(),
            truth=_unmatched_truth(),
            spread_thresholds_m=(0.0,),
            fallback_blends=(0.0, 0.5),
        )


def test_blend_search_rejects_before_creating_output_directory(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    output_dir = tmp_path / "out"
    _estimate_rows().to_csv(estimate_csv, index=False)

    with pytest.raises(
        ValueError,
        match="no spread-guard blend candidate had finite matched truth rows",
    ):
        write_spread_guard_blend_search_outputs(
            estimate_inputs=[EstimateInput("estimate", estimate_csv)],
            template=_template(),
            truth=_unmatched_truth(),
            output_dir=output_dir,
            spread_thresholds_m=(0.0,),
            fallback_blends=(0.0, 0.5),
        )

    assert not output_dir.exists()
