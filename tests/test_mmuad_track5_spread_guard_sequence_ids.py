from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import write_spread_guard_outputs


def test_track5_spread_guard_estimate_csv_preserves_zero_padded_sequence_id(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "trusted.csv"
    pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    ).to_csv(estimate_csv, index=False)

    paths = write_spread_guard_outputs(
        estimate_inputs=[parse_estimate_spec(f"trusted={estimate_csv}@1.0")],
        template=pd.DataFrame(
            {
                "Sequence": ["001"],
                "Timestamp": [0.0],
                "Position": ["(0,0,0)"],
                "Classification": [2],
            }
        ),
        output_dir=tmp_path / "out",
        spread_threshold_m=0.0,
        class_map={"001": "2"},
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    estimates = pd.read_csv(
        paths["estimates_csv"], dtype={"sequence_id": str}, keep_default_na=False
    )
    official = pd.read_csv(
        paths["official_results_csv"], dtype={"Sequence": str}, keep_default_na=False
    )
    assert estimates["sequence_id"].tolist() == ["001"]
    assert official["Sequence"].tolist() == ["001"]


def test_track5_spread_guard_estimate_csv_strips_padded_sequence_headers(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "trusted.csv"
    estimate_csv.write_text(
        " Sequence , Timestamp , state_x_m , state_y_m , state_z_m \n"
        "001,0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )

    paths = write_spread_guard_outputs(
        estimate_inputs=[parse_estimate_spec(f"trusted={estimate_csv}@1.0")],
        template=pd.DataFrame(
            {
                "Sequence": ["001"],
                "Timestamp": [0.0],
                "Position": ["(0,0,0)"],
                "Classification": [2],
            }
        ),
        output_dir=tmp_path / "out",
        spread_threshold_m=0.0,
        class_map={"001": "2"},
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    estimates = pd.read_csv(
        paths["estimates_csv"], dtype={"sequence_id": str}, keep_default_na=False
    )
    official = pd.read_csv(
        paths["official_results_csv"], dtype={"Sequence": str}, keep_default_na=False
    )
    assert estimates["sequence_id"].tolist() == ["001"]
    assert official["Sequence"].tolist() == ["001"]
