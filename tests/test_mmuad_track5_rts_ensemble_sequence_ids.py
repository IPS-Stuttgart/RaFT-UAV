from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_rts_ensemble import (
    build_track5_rts_ensemble,
    write_track5_rts_ensemble_outputs,
)


def test_rts_ensemble_accepts_padded_template_headers_and_ids() -> None:
    template = pd.DataFrame(
        {
            " Sequence ": [" 001 ", "001"],
            " Timestamp ": [0.0, 1.0],
            " Position ": ["(0,0,0)", "(0,0,0)"],
            " Classification ": [2, 2],
        }
    )
    estimates = pd.DataFrame(
        {
            "sequence_id": ["001", "001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
        }
    )

    smoothed, diagnostics = build_track5_rts_ensemble(
        [("estimate", estimates, 1.0)],
        template,
        measurement_sigma_m=1.0,
        process_accel_std_mps2=0.1,
    )

    assert smoothed["sequence_id"].tolist() == ["001", "001"]
    assert diagnostics["valid_input_count"].tolist() == [1, 1]
    assert diagnostics["weighted_x_m"].tolist() == pytest.approx([1.0, 2.0])


def test_rts_ensemble_output_reader_preserves_zero_padded_sequence_ids(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    estimate_csv.write_text(
        "Sequence,Timestamp,x,y,z\n"
        "001,0.0,1.0,3.0,5.0\n"
        "001,1.0,2.0,4.0,6.0\n",
        encoding="utf-8",
    )
    template = pd.DataFrame(
        {
            "Sequence": ["001", "001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2],
        }
    )

    paths = write_track5_rts_ensemble_outputs(
        estimate_inputs=[parse_estimate_spec(f"estimate={estimate_csv}@1.0")],
        template=template,
        output_dir=tmp_path / "out",
        class_map={"001": "2"},
        measurement_sigma_m=1.0,
        process_accel_std_mps2=0.1,
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    official = pd.read_csv(paths["official_results_csv"], dtype=str)
    diagnostics = pd.read_csv(paths["diagnostics_csv"], dtype={"sequence_id": str})

    assert manifest["valid_rows"] == 2
    assert official["Sequence"].tolist() == ["001", "001"]
    assert official["Classification"].tolist() == ["2", "2"]
    assert diagnostics["sequence_id"].tolist() == ["001", "001"]
    assert diagnostics["valid_input_count"].tolist() == [1, 1]
