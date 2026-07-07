from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble_grid import evaluate_estimate_ensemble_weight_grid
from raft_uav.mmuad.track5_estimate_ensemble_grid import generate_simplex_weight_grid


def test_estimate_ensemble_grid_preserves_zero_padded_sequence_ids(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(good, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["1"],
            "time_s": [0.0],
            "state_x_m": [100.0],
            "state_y_m": [100.0],
            "state_z_m": [100.0],
        }
    ).to_csv(bad, index=False)
    template = pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
            "class_name": ["2"],
        }
    )

    summary, by_sequence, best_weights = evaluate_estimate_ensemble_weight_grid(
        [parse_estimate_spec(f"good={good}"), parse_estimate_spec(f"bad={bad}")],
        template=template,
        truth=truth,
        weight_grid=generate_simplex_weight_grid(2, step=0.5),
        default_classification=2,
    )

    assert best_weights == (1.0, 0.0)
    assert summary.iloc[0]["weight_good"] == pytest.approx(1.0)
    assert summary.iloc[0]["pose_mse"] == pytest.approx(0.0)
    assert set(by_sequence["sequence_id"]) == {"001"}
