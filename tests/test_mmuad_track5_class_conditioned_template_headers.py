from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_class_conditioned_ensemble import (
    build_class_conditioned_estimate_ensemble,
    search_class_conditioned_ensemble_weights,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


def test_class_conditioned_ensemble_accepts_padded_template_headers(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001", "001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
        }
    ).to_csv(estimate_csv, index=False)
    template = pd.DataFrame(
        {
            " Sequence ": ["001", "001"],
            " Timestamp ": [0.0, 1.0],
            " Position ": ["(0,0,0)", "(0,0,0)"],
            " Classification ": [0, 0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["001", "001"],
            "time_s": [0.0, 1.0],
            "x_m": [1.0, 2.0],
            "y_m": [3.0, 4.0],
            "z_m": [5.0, 6.0],
        }
    )
    estimate_input = EstimateInput("estimate", estimate_csv)

    _, config = search_class_conditioned_ensemble_weights(
        [estimate_input],
        template=template,
        truth=truth,
        class_map={"001": "0"},
        weight_step=0.5,
    )
    estimates, diagnostics = build_class_conditioned_estimate_ensemble(
        [estimate_input],
        template=template,
        class_map={"001": "0"},
        weight_config=config,
    )

    assert config["class_weights"]["0"] == {"estimate": 1.0}
    assert config["metrics"]["0"]["pose_mse_m2"] == pytest.approx(0.0)
    assert estimates["sequence_id"].tolist() == ["001", "001"]
    assert estimates["state_x_m"].tolist() == pytest.approx([1.0, 2.0])
    assert diagnostics["valid_input_count"].tolist() == [1, 1]
