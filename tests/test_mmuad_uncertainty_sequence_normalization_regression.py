from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_ensemble import build_track5_uncertainty_ensemble


def test_uncertainty_ensemble_normalizes_template_and_sigma_sequences(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 2.0],
            "state_x_m": [0.0, 2.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [5.0, 5.0],
            "predicted_sigma_m": [2.0, 2.0],
        }
    ).to_csv(estimate_csv, index=False)
    template = pd.DataFrame(
        {
            "Sequence": [" seq0001 "],
            "Timestamp": [1.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("base", estimate_csv, 1.0)],
        template=template,
        uncertainty_column="predicted_sigma_m",
    )

    row = estimates.iloc[0]
    assert row["sequence_id"] == "seq0001"
    assert row["ensemble_source_count"] == 1
    assert row["state_x_m"] == pytest.approx(1.0)
    assert row["state_y_m"] == pytest.approx(0.0)
    assert row["state_z_m"] == pytest.approx(5.0)
    assert row["ensemble_effective_sigma_m"] == pytest.approx(2.0)
    assert diagnostics.iloc[0]["valid_input_count"] == 1
