from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_class_conditioned_ensemble import (
    build_class_conditioned_estimate_ensemble,
    search_class_conditioned_ensemble_weights,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


def test_class_conditioned_search_and_apply_preserve_zero_padded_csv_sequence_ids(
    tmp_path: Path,
) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(good_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [101.0],
            "state_y_m": [102.0],
            "state_z_m": [103.0],
        }
    ).to_csv(bad_csv, index=False)
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
        }
    )

    assert pd.read_csv(good_csv)["sequence_id"].iloc[0] == 1

    inputs = [EstimateInput("good", good_csv), EstimateInput("bad", bad_csv)]
    _, config = search_class_conditioned_ensemble_weights(
        inputs,
        template=template,
        truth=truth,
        class_map={"001": "2"},
        weight_step=1.0,
    )

    assert config["class_weights"]["2"] == {"good": 1.0, "bad": 0.0}
    assert config["metrics"]["2"]["pose_mse_m2"] == pytest.approx(0.0)

    estimates, diagnostics = build_class_conditioned_estimate_ensemble(
        inputs,
        template=template,
        class_map={"001": "2"},
        weight_config=config,
    )

    assert estimates.loc[0, "sequence_id"] == "001"
    assert estimates.loc[0, "state_x_m"] == pytest.approx(1.0)
    assert estimates.loc[0, "state_y_m"] == pytest.approx(2.0)
    assert estimates.loc[0, "state_z_m"] == pytest.approx(3.0)
    assert diagnostics.loc[0, "valid_input_count"] == 2
