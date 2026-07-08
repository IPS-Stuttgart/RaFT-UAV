from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_sequence_gate import main as sequence_gate_main


def test_sequence_gate_cli_accepts_padded_weight_and_template_headers(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    alternate_path = tmp_path / "alternate.csv"
    weights_path = tmp_path / "weights.csv"
    template_path = tmp_path / "template.csv"
    output_dir = tmp_path / "out"

    base_path.write_text(
        'Sequence,Timestamp,Position,Classification\n001,0.0,"(0.0,0.0,0.0)",1\n',
        encoding="utf-8",
    )
    alternate_path.write_text(
        'Sequence,Timestamp,Position,Classification\n001,0.0,"(4.0,0.0,0.0)",1\n',
        encoding="utf-8",
    )
    weights_path.write_text(" Sequence , weight \n001,1.0\n", encoding="utf-8")
    template_path.write_text(
        ' Sequence , Timestamp , Position , Classification \n001,0.0,"(0.0,0.0,0.0)",1\n',
        encoding="utf-8",
    )

    status = sequence_gate_main(
        [
            "--base-submission",
            str(base_path),
            "--alternate-submission",
            str(alternate_path),
            "--sequence-weights",
            str(weights_path),
            "--template",
            str(template_path),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    estimates = pd.read_csv(
        output_dir / "mmuad_track5_sequence_gate_estimates.csv",
        dtype={"sequence_id": "string"},
    )
    diagnostics = pd.read_csv(
        output_dir / "mmuad_track5_sequence_gate_diagnostics.csv",
        dtype={"sequence_id": "string"},
    )
    manifest = json.loads(
        (output_dir / "mmuad_track5_sequence_gate_manifest.json").read_text(encoding="utf-8")
    )

    assert estimates.loc[0, "sequence_id"] == "001"
    assert estimates.loc[0, "state_x_m"] == pytest.approx(4.0)
    assert diagnostics.loc[0, "weight_source"] == "sequence_weights"
    assert manifest["validation"]["leaderboard_ready"] is True
    assert manifest["defaulted_sequence_count"] == 0
