from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_estimate_calibration import fit_track5_estimate_calibration
from raft_uav.mmuad.track5_estimate_calibration_shrinkage import main as shrinkage_main


def test_calibration_shrinkage_cli_preserves_zero_padded_sequence_ids(tmp_path: Path) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_csv = tmp_path / "template.csv"
    truth_csv = tmp_path / "truth.csv"
    calibration_json = tmp_path / "calibration.json"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"

    estimates = pd.DataFrame(
        {
            "Sequence": ["001", "001"],
            "Timestamp": ["0.0", "1.0"],
            "state_x_m": ["1.0", "2.0"],
            "state_y_m": ["3.0", "4.0"],
            "state_z_m": ["5.0", "6.0"],
        }
    )
    template = pd.DataFrame(
        {
            "Sequence": ["001", "001"],
            "Timestamp": ["0.0", "1.0"],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": ["2", "2"],
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

    estimates.to_csv(estimates_csv, index=False)
    template.to_csv(template_csv, index=False)
    truth.to_csv(truth_csv, index=False)
    class_map_csv.write_text("sequence_id,uav_type\n001,2\n", encoding="utf-8")

    calibration, _ = fit_track5_estimate_calibration(
        estimates,
        template=template,
        truth=truth,
        mode="identity",
    )
    calibration_json.write_text(json.dumps(calibration), encoding="utf-8")

    status = shrinkage_main(
        [
            "--estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_csv),
            "--calibration-json",
            str(calibration_json),
            "--truth-csv",
            str(truth_csv),
            "--alpha-grid",
            "1",
            "--use-best-alpha",
            "--write-apply",
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    manifest = json.loads(
        (output_dir / "apply" / "mmuad_track5_calibration_shrinkage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["valid_shrunk_rows"] == 2
    assert manifest["leaderboard_ready"] is True

    diagnostics = pd.read_csv(
        output_dir / "apply" / "mmuad_track5_calibration_shrinkage_diagnostics.csv",
        dtype={"sequence_id": str},
    )
    assert diagnostics["sequence_id"].astype(str).tolist() == ["001", "001"]
