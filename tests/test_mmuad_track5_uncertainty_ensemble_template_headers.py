from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_ensemble import build_track5_uncertainty_ensemble


def test_uncertainty_ensemble_accepts_padded_template_headers(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            "predicted_sigma_m": [1.0],
        }
    ).to_csv(estimate_csv, index=False)
    template = pd.DataFrame(
        {
            " Sequence ": ["001"],
            " Timestamp ": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("estimate", estimate_csv, 1.0)],
        template=template,
    )

    row = estimates.iloc[0]
    assert row["sequence_id"] == "001"
    assert row["state_x_m"] == pytest.approx(1.0)
    assert row["state_y_m"] == pytest.approx(2.0)
    assert row["state_z_m"] == pytest.approx(3.0)
    assert row["ensemble_source_count"] == 1
    assert diagnostics["valid_input_count"].tolist() == [1]
