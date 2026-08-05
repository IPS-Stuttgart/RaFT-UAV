from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_rts_ensemble_grid import run_track5_rts_ensemble_grid_search


def test_rts_ensemble_grid_materializes_one_shot_iterables(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 1.0],
        }
    ).to_csv(estimate_csv, index=False)
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 3,
            "Timestamp": [0.0, 1.0, 2.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 2],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )
    measurement_values = (5.0, 10.0)
    process_values = (1.0, 3.0)
    spread_values = (0.0, 1.0)

    grid, _ = run_track5_rts_ensemble_grid_search(
        [parse_estimate_spec(f"base={estimate_csv}")],
        template=template,
        truth=truth,
        measurement_sigma_grid=(value for value in measurement_values),
        process_accel_grid=(value for value in process_values),
        spread_variance_scale_grid=(value for value in spread_values),
        score_time_tolerance_s=1.0e-9,
    )

    actual = set(
        grid[
            [
                "measurement_sigma_m",
                "process_accel_std_mps2",
                "spread_variance_scale",
            ]
        ].itertuples(index=False, name=None)
    )
    assert actual == set(product(measurement_values, process_values, spread_values))
