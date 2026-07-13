from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_rts_ensemble_grid import run_track5_rts_ensemble_grid_search


def test_rts_ensemble_grid_materializes_one_shot_parameter_grids(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 8.0, 3.0, 4.0],
            "state_y_m": [0.0] * 5,
            "state_z_m": [1.0] * 5,
        }
    ).to_csv(estimate_csv, index=False)
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 5,
            "Timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Position": ["(0,0,0)"] * 5,
            "Classification": [2] * 5,
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y_m": [0.0] * 5,
            "z_m": [1.0] * 5,
        }
    )
    measurement_sigmas = (value for value in (5.0, 10.0))
    process_accels = (value for value in (0.5, 5.0))
    spread_scales = (value for value in (0.0, 1.0))

    grid, _ = run_track5_rts_ensemble_grid_search(
        [parse_estimate_spec(f"base={estimate_csv}")],
        template=template,
        truth=truth,
        measurement_sigma_grid=measurement_sigmas,
        process_accel_grid=process_accels,
        spread_variance_scale_grid=spread_scales,
        score_time_tolerance_s=1.0e-9,
    )

    actual = {
        tuple(row)
        for row in grid[
            [
                "measurement_sigma_m",
                "process_accel_std_mps2",
                "spread_variance_scale",
            ]
        ].itertuples(index=False, name=None)
    }
    expected = set(product((5.0, 10.0), (0.5, 5.0), (0.0, 1.0)))
    assert actual == expected
