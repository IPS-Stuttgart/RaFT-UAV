from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_consensus_ensemble_grid import (
    search_track5_consensus_ensemble_grid,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


def test_consensus_grid_reuses_generators_for_full_cartesian_product(
    tmp_path: Path,
) -> None:
    estimate_path = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    ).to_csv(estimate_path, index=False)
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )
    radius_values = (1.0, 2.0)
    fraction_values = (0.0, 0.5)
    fallback_values = ("max-weight", "weighted-mean")

    grid, _ = search_track5_consensus_ensemble_grid(
        [EstimateInput("base", estimate_path, 1.0)],
        template=template,
        truth=truth,
        consensus_radius_m=(value for value in radius_values),
        min_consensus_weight_fraction=(value for value in fraction_values),
        fallback_policy=(value for value in fallback_values),
    )

    actual = set(
        grid[
            [
                "consensus_radius_m",
                "min_consensus_weight_fraction",
                "fallback_policy",
            ]
        ].itertuples(index=False, name=None)
    )
    expected = set(product(radius_values, fraction_values, fallback_values))
    assert len(grid) == len(expected) == 8
    assert actual == expected
