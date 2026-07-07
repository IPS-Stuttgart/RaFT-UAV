from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.submission import write_official_mmaud_results_csv
from raft_uav.mmuad.track5_acceleration_limit import main as acceleration_main
from raft_uav.mmuad.track5_acceleration_limit import project_track5_acceleration_limit
from raft_uav.mmuad.track5_acceleration_limit import write_track5_acceleration_limit_outputs
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 4,
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "state_x_m": [0.0, 0.0, 100.0, 101.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
            "Classification": [2, 2, 2, 2],
        }
    )


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 4,
            "Timestamp": [0.0, 1.0, 2.0, 3.0],
            "Position": ["(0,0,0)"] * 4,
            "Classification": [2, 2, 2, 2],
        }
    )


def test_acceleration_limit_reduces_short_acceleration_spike() -> None:
    limited, diagnostics = project_track5_acceleration_limit(
        _submission_rows(),
        max_acceleration_mps2=2.0,
        iterations=2,
    )

    assert diagnostics["acceleration_limit_applied"].any()
    assert limited.loc[2, "state_x_m"] < 100.0
    assert diagnostics["limited_acceleration_mps2"].max() < diagnostics[
        "input_acceleration_mps2"
    ].max()
    assert limited["Classification"].tolist() == [2, 2, 2, 2]


def test_acceleration_limit_writes_leaderboard_ready_outputs(tmp_path: Path) -> None:
    paths = write_track5_acceleration_limit_outputs(
        limited=project_track5_acceleration_limit(_submission_rows(), max_acceleration_mps2=2.0)[0],
        diagnostics=project_track5_acceleration_limit(_submission_rows(), max_acceleration_mps2=2.0)[1],
        output_dir=tmp_path,
        input_submission_path=tmp_path / "input.csv",
        template=_template_rows(),
        require_leaderboard_ready=True,
    )

    assert paths["zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["changed_row_count"] > 0
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_acceleration_limit_cli_reads_official_submission(tmp_path: Path) -> None:
    submission_csv = tmp_path / "mmaud_results.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    write_official_mmaud_results_csv(
        _submission_rows(),
        submission_csv,
        classification=2,
        invalid_row_policy="raise",
    )
    _template_rows().to_csv(template_csv, index=False)

    status = acceleration_main(
        [
            "--submission",
            str(submission_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--max-acceleration-mps2",
            "2",
            "--iterations",
            "2",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    limited = load_track5_submission(output_dir / "ug2_submission_acceleration_limited.zip")
    assert limited["Classification"].tolist() == [2, 2, 2, 2]
    assert (output_dir / "mmuad_track5_acceleration_limit_manifest.json").exists()


def test_acceleration_limit_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-acceleration-limit"]
        == "raft_uav.mmuad.track5_acceleration_limit:main"
    )
