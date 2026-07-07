from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad import track5_template_resample_cli as cli


def test_template_resample_cli_uses_hardened_estimate_csv_reader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    estimates_csv.write_text(
        " Sequence , Timestamp , state_x_m , state_y_m , state_z_m \n"
        "001,0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    template_csv.write_text("unused by monkeypatched template loader\n", encoding="utf-8")

    observed: dict[str, Any] = {}

    def fake_template_loader(path: Path) -> pd.DataFrame:
        observed["template_path"] = path
        return pd.DataFrame({"Sequence": ["001"], "Timestamp": ["0.0"]})

    def fake_writer(**kwargs: Any) -> dict[str, Path]:
        estimates = kwargs["estimates"]
        observed["estimate_columns"] = list(estimates.columns)
        observed["sequence_value"] = estimates.loc[0, "Sequence"]
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_json = output_dir / "validation.json"
        validation_json.write_text(
            json.dumps({"leaderboard_ready": True, "codabench_upload_ready": True}),
            encoding="utf-8",
        )
        return {"validation_json": validation_json}

    monkeypatch.setattr(cli, "load_official_track5_template_file", fake_template_loader)
    monkeypatch.setattr(cli, "write_track5_template_resample_outputs", fake_writer)

    status = cli.main(
        [
            "--estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    assert observed["template_path"] == template_csv
    assert observed["estimate_columns"] == [
        "Sequence",
        "Timestamp",
        "state_x_m",
        "state_y_m",
        "state_z_m",
    ]
    assert observed["sequence_value"] == "001"
