from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble_weight_search import (
    search_track5_estimate_ensemble_weights,
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


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    complete_path = tmp_path / "complete.csv"
    incomplete_path = tmp_path / "incomplete.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [1.0, 11.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    ).to_csv(incomplete_path, index=False)
    return complete_path, incomplete_path


@pytest.mark.parametrize(
    "selection_objective",
    [
        "pooled-mse",
        "mean-sequence-mse",
        "max-sequence-mse",
        "pooled-plus-max-sequence-mse",
    ],
)
def test_weight_search_does_not_reward_missing_truth_rows(
    tmp_path: Path,
    selection_objective: str,
) -> None:
    complete_path, incomplete_path = _write_inputs(tmp_path)

    grid, best = search_track5_estimate_ensemble_weights(
        [
            EstimateInput("complete", complete_path),
            EstimateInput("incomplete", incomplete_path),
        ],
        template=_template(),
        truth=_truth(),
        weight_step=1.0,
        max_nearest_time_delta_s=0.0,
        selection_objective=selection_objective,
    )

    incomplete = grid.loc[grid["weight_incomplete"] == 1.0].iloc[0]
    assert incomplete["matched_rows"] == 1
    assert incomplete["truth_rows"] == 2
    assert incomplete["unmatched_rows"] == 1
    assert incomplete["coverage_fraction"] == pytest.approx(0.5)
    assert np.isinf(incomplete["selection_objective_value"])

    assert best["weights"] == {"complete": 1.0, "incomplete": 0.0}
    assert best["metrics"]["matched_rows"] == 2
