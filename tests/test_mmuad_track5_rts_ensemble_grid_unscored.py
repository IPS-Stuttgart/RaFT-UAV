from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_rts_ensemble_grid import run_track5_rts_ensemble_grid_search
from raft_uav.mmuad.track5_rts_ensemble_grid import write_track5_rts_ensemble_grid_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 3,
            "Timestamp": [0.0, 1.0, 2.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2] * 3,
        }
    )


def _truth_without_overlap() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq9999"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0] * 3,
            "z_m": [1.0] * 3,
        }
    )


def _estimate_input(tmp_path: Path):
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0] * 3,
            "state_z_m": [1.0] * 3,
        }
    ).to_csv(estimate_csv, index=False)
    return parse_estimate_spec(f"base={estimate_csv}")


def test_rts_ensemble_grid_rejects_unscored_candidates(tmp_path: Path) -> None:
    estimate_input = _estimate_input(tmp_path)

    with pytest.raises(ValueError, match="no scored candidates"):
        run_track5_rts_ensemble_grid_search(
            [estimate_input],
            template=_template(),
            truth=_truth_without_overlap(),
            measurement_sigma_grid=(5.0,),
            process_accel_grid=(1.0,),
            spread_variance_scale_grid=(0.0,),
        )


def test_rts_ensemble_grid_fails_before_creating_output_directory(tmp_path: Path) -> None:
    estimate_input = _estimate_input(tmp_path)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="no scored candidates"):
        write_track5_rts_ensemble_grid_outputs(
            estimate_inputs=[estimate_input],
            template=_template(),
            truth=_truth_without_overlap(),
            output_dir=output_dir,
            measurement_sigma_grid=(5.0,),
            process_accel_grid=(1.0,),
            spread_variance_scale_grid=(0.0,),
        )

    assert not output_dir.exists()
