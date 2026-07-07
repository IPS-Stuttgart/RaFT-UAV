from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_soft_class_ensemble import main as soft_class_main


def _weight_config() -> dict[str, object]:
    return {
        "schema": "test-class-conditioned",
        "aggregation_policy": "weighted-mean",
        "trim_fraction": 0.2,
        "global_weights": {"a": 1.0, "b": 0.0},
        "class_weights": {
            "0": {"a": 1.0, "b": 0.0},
            "1": {"a": 0.0, "b": 1.0},
        },
    }


def test_soft_class_ensemble_cli_preserves_numeric_like_sequence_ids(tmp_path: Path) -> None:
    a_csv = tmp_path / "a_numeric.csv"
    b_csv = tmp_path / "b_numeric.csv"
    template_csv = tmp_path / "template_numeric.csv"
    probs_csv = tmp_path / "probabilities_numeric.csv"
    weights_json = tmp_path / "weights_numeric.json"
    class_map_csv = tmp_path / "class_map_numeric.csv"
    output_dir = tmp_path / "out_numeric"

    pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    ).to_csv(template_csv, index=False)
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    ).to_csv(a_csv, index=False)
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "state_x_m": [100.0],
            "state_y_m": [100.0],
            "state_z_m": [100.0],
        }
    ).to_csv(b_csv, index=False)
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "predicted_probability_0": [0.0],
            "predicted_probability_1": [1.0],
            "predicted_probability_2": [0.0],
            "predicted_probability_3": [0.0],
        }
    ).to_csv(probs_csv, index=False)
    weights_json.write_text(json.dumps(_weight_config()), encoding="utf-8")
    pd.DataFrame({"sequence_id": ["001"], "uav_type": [1]}).to_csv(class_map_csv, index=False)

    status = soft_class_main(
        [
            "--estimate-csv",
            f"a={a_csv}",
            "--estimate-csv",
            f"b={b_csv}",
            "--template",
            str(template_csv),
            "--class-probabilities-csv",
            str(probs_csv),
            "--weight-config-json",
            str(weights_json),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    estimates = pd.read_csv(
        output_dir / "mmuad_track5_soft_class_ensemble_estimates.csv",
        dtype={"sequence_id": str},
    )
    row = estimates.iloc[0]

    assert status == 0
    assert row["sequence_id"] == "001"
    assert row["state_x_m"] == pytest.approx(100.0)
    assert row["state_y_m"] == pytest.approx(100.0)
    assert str(row["soft_class_probability_available"]).lower() == "true"
