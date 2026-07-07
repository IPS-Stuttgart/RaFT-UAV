from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble_grid import evaluate_estimate_ensemble_weight_grid


def test_estimate_ensemble_weight_grid_preserves_zero_padded_sequence_ids(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(estimate_csv, index=False)

    # This is the failure mode being guarded: plain pandas CSV loading turns
    # opaque Track 5 sequence ids into integers before template resampling.
    assert pd.read_csv(estimate_csv)["sequence_id"].iloc[0] == 1

    summary, by_sequence, best_weights = evaluate_estimate_ensemble_weight_grid(
        [parse_estimate_spec(f"model={estimate_csv}")],
        template=pd.DataFrame(
            {
                "Sequence": ["001"],
                "Timestamp": [0.0],
                "Position": ["(0,0,0)"],
                "Classification": [2],
            }
        ),
        truth=pd.DataFrame(
            {
                "sequence_id": ["001"],
                "time_s": [0.0],
                "x_m": [1.0],
                "y_m": [2.0],
                "z_m": [3.0],
                "class_name": ["2"],
            }
        ),
        weight_grid=[(1.0,)],
        default_classification=2,
    )

    assert best_weights == (1.0,)
    assert summary.iloc[0]["pose_mse"] == pytest.approx(0.0)
    assert set(by_sequence["sequence_id"]) == {"001"}
