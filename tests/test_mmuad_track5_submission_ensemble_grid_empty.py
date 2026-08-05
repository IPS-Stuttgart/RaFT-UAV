from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import parse_submission_input
from raft_uav.mmuad.track5_submission_ensemble_grid import (
    evaluate_submission_ensemble_weight_grid,
    generate_simplex_weight_grid,
)


def test_submission_ensemble_grid_rejects_empty_weight_grid(tmp_path: Path) -> None:
    submission = tmp_path / "only.csv"
    pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0.0, 0.0, 1.0)"],
            "Classification": [1],
        }
    ).to_csv(submission, index=False)
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
            "class_id": [1],
        }
    )
    empty_grid = generate_simplex_weight_grid(
        1,
        step=0.5,
        include_singletons=False,
    )

    with pytest.raises(ValueError, match="weight grid produced no rows"):
        evaluate_submission_ensemble_weight_grid(
            [parse_submission_input(f"only={submission}")],
            truth=truth,
            weight_grid=empty_grid,
        )
