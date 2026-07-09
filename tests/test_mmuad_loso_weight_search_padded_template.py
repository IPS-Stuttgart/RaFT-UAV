from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble_loso_weight_search import (
    run_track5_estimate_ensemble_loso_weight_search,
)


def test_loso_weight_search_accepts_padded_template_headers(tmp_path: Path) -> None:
    template = pd.DataFrame(
        {
            " Sequence ": ["001", "001", "002", "002"],
            " Timestamp ": [0.0, 1.0, 0.0, 1.0],
            " Position ": ["(0,0,0)"] * 4,
            " Classification ": [2, 2, 1, 1],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["001", "001", "002", "002"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "x_m": [0.0, 1.0, 4.0, 5.0],
            "y_m": [0.0, 0.0, 4.0, 4.0],
            "z_m": [0.0, 0.0, 4.0, 4.0],
        }
    )
    estimates = pd.DataFrame(
        {
            "sequence_id": ["001", "001", "002", "002"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "state_x_m": [0.0, 1.0, 4.0, 5.0],
            "state_y_m": [0.0, 0.0, 4.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0, 4.0],
        }
    )
    estimates_csv = tmp_path / "estimate.csv"
    estimates.to_csv(estimates_csv, index=False)

    folds, predictions, summary, _ = run_track5_estimate_ensemble_loso_weight_search(
        [EstimateInput("estimate", estimates_csv)],
        template=template,
        truth=truth,
        weight_step=1.0,
    )

    assert int(folds["matched_rows"].sum()) == 4
    assert summary["loso_metrics"]["matched_rows"] == 4
    assert summary["loso_metrics"]["pose_mse_m2"] == pytest.approx(0.0)
    assert len(predictions) == 4
