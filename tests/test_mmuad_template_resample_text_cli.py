from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_template_resample_cli import main as resample_main


def test_template_resample_cli_keeps_zero_padded_sequence_ids(tmp_path: Path) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"

    pd.DataFrame(
        {
            "Sequence": ["001", "001"],
            "Timestamp": ["0.0", "10.0"],
            "state_x_m": ["1.0", "11.0"],
            "state_y_m": ["2.0", "12.0"],
            "state_z_m": ["3.0", "13.0"],
            "Classification": ["4", "4"],
        }
    ).to_csv(estimates_csv, index=False)
    pd.DataFrame(
        {
            "Sequence": ["001", "001"],
            "Timestamp": ["0.0", "10.0"],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": ["4", "4"],
        }
    ).to_csv(template_csv, index=False)

    status = resample_main(
        [
            "--estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    manifest = json.loads((output_dir / "mmuad_template_resample_manifest.json").read_text())
    assert manifest["invalid_sequence_count"] == 0
    assert manifest["leaderboard_ready"] is True
    diagnostics = pd.read_csv(
        output_dir / "mmuad_template_resample_diagnostics_by_sequence.csv",
        dtype={"sequence_id": str},
    )
    assert diagnostics["sequence_id"].astype(str).tolist() == ["001"]
    assert diagnostics["valid_row_count"].tolist() == [2]
