from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_ensemble import build_track5_uncertainty_ensemble


def test_uncertainty_ensemble_strips_header_whitespace_from_estimate_csv(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    estimate_csv.write_text(
        " sequence_id , time_s , state_x_m , state_y_m , state_z_m , predicted_sigma_m \n"
        "001,0.0,1.0,2.0,3.0,1.0\n",
        encoding="utf-8",
    )
    template = pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
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
