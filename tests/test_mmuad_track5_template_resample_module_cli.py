from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

import raft_uav.mmuad.track5_template_resample_cli as template_resample_cli


def test_module_cli_preserves_zero_padded_sequence_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_path = tmp_path / "template.csv"
    output_dir = tmp_path / "output"
    validation_json = tmp_path / "validation.json"
    estimates_csv.write_text(
        "sequence_id,time_s,state_x_m,state_y_m,state_z_m\n"
        "001,0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    template_path.write_text("unused\n", encoding="utf-8")
    validation_json.write_text(
        '{"leaderboard_ready": true, "codabench_upload_ready": true}',
        encoding="utf-8",
    )
    captured: dict[str, pd.DataFrame] = {}

    def fake_load_template(_path: Path) -> pd.DataFrame:
        return pd.DataFrame({"Sequence": ["001"], "Timestamp": [0.0]})

    def fake_write_outputs(**kwargs: object) -> dict[str, Path]:
        captured["estimates"] = pd.DataFrame(kwargs["estimates"]).copy()
        return {"validation_json": validation_json}

    monkeypatch.setattr(
        template_resample_cli,
        "load_official_track5_template_file",
        fake_load_template,
    )
    monkeypatch.setattr(
        template_resample_cli,
        "write_track5_template_resample_outputs",
        fake_write_outputs,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m raft_uav.mmuad.track5_template_resample",
            "--estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(
            "raft_uav.mmuad.track5_template_resample.__main__",
            run_name="__main__",
        )

    assert exc_info.value.code == 0
    assert captured["estimates"].loc[0, "sequence_id"] == "001"
