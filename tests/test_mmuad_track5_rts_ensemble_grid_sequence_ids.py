from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_rts_ensemble_grid import run_track5_rts_ensemble_grid_search


def test_rts_grid_preserves_zero_padded_sequence_ids_from_estimate_csv(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    estimate_csv.write_text(
        "sequence_id,time_s,state_x_m,state_y_m,state_z_m\n"
        "001,0.0,0.0,0.0,1.0\n"
        "001,1.0,1.0,0.0,1.0\n",
        encoding="utf-8",
    )
    template = pd.DataFrame(
        {
            "Sequence": ["001", "001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,1)", "(1,0,1)"],
            "Classification": [2, 2],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["001", "001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    )

    grid, best = run_track5_rts_ensemble_grid_search(
        [parse_estimate_spec(f"estimate={estimate_csv}")],
        template=template,
        truth=truth,
        measurement_sigma_grid=(1.0,),
        process_accel_grid=(1.0,),
        spread_variance_scale_grid=(0.0,),
        score_time_tolerance_s=1.0e-9,
    )

    assert grid.loc[0, "matched_row_count"] == 2
    assert grid.loc[0, "diagnostic_valid_input_count_mean"] == 1.0
    assert np.isfinite(grid.loc[0, "pose_mse_m2"])
    assert best["best"]["matched_row_count"] == 2
