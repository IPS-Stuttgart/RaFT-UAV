from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_ensemble_weight_search import (
    search_track5_uncertainty_ensemble_weights,
)


def test_uncertainty_weight_search_matches_integer_truth_timestamps(tmp_path: Path) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [0.0, 1.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
            "predicted_sigma_m": [5.0, 5.0, 5.0],
        }
    ).to_csv(estimates_csv, index=False)
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0, 1, 0],
            "x_m": [0.0, 1.0, 4.0],
            "y_m": [0.0, 0.0, 4.0],
            "z_m": [0.0, 0.0, 4.0],
        }
    )

    assert pd.api.types.is_integer_dtype(truth["time_s"])

    grid, best = search_track5_uncertainty_ensemble_weights(
        [EstimateInput("estimate", estimates_csv)],
        template=template,
        truth=truth,
        uncertainty_column="predicted_sigma_m",
        weight_step=1.0,
    )

    assert grid["matched_rows"].tolist() == [3]
    assert best["metrics"]["pose_mse_m2"] == pytest.approx(0.0)
