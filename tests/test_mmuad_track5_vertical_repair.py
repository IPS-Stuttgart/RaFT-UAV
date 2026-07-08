from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import main as vertical_repair_main
from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes
from raft_uav.mmuad.track5_vertical_repair import write_track5_vertical_repair_outputs


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "state_z_m": [10.0, 11.0, 80.0, 13.0, 14.0],
            "Classification": [2, 2, 2, 2, 2],
        }
    )


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 5,
            "Timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Position": ["(0,0,0)"] * 5,
            "Classification": [2, 2, 2, 2, 2],
        }
    )


def test_vertical_repair_changes_only_isolated_z_spike() -> None:
    repaired, diagnostics = repair_track5_vertical_spikes(
        _submission_rows(),
        max_vertical_speed_mps=20.0,
        max_neighbor_vertical_speed_mps=10.0,
        max_vertical_residual_m=15.0,
        max_horizontal_speed_mps=20.0,
        iterations=2,
    )

    repaired_row = repaired.loc[repaired["time_s"] == 2.0].iloc[0]
    assert repaired_row["state_z_m"] == pytest.approx(12.0)
    assert repaired_row["state_x_m"] == pytest.approx(2.0)
    assert repaired_row["state_y_m"] == pytest.approx(0.0)
    assert repaired_row["Classification"] == 2
    assert int(diagnostics["repaired"].sum()) == 1
    repaired_delta = diagnostics.loc[
        diagnostics["repaired"],
        "vertical_repair_m",
    ].iloc[0]
    assert repaired_delta == pytest.approx(-68.0)


def test_vertical_repair_keeps_point_when_horizontal_motion_is_implausible() -> None:
    rows = _submission_rows()
    rows.loc[2, "state_x_m"] = 1000.0
    repaired, diagnostics = repair_track5_vertical_spikes(
        rows,
        max_vertical_speed_mps=20.0,
        max_neighbor_vertical_speed_mps=10.0,
        max_vertical_residual_m=15.0,
        max_horizontal_speed_mps=20.0,
        iterations=1,
    )

    kept_z = repaired.loc[repaired["time_s"] == 2.0, "state_z_m"].iloc[0]
    assert kept_z == pytest.approx(80.0)
    assert int(diagnostics["repaired"].sum()) == 0
    horizontal_ok = diagnostics.loc[
        diagnostics["time_s"] == 2.0,
        "horizontal_gate_ok",
    ].iloc[0]
    assert not horizontal_ok


def test_vertical_repair_writes_leaderboard_ready_outputs(tmp_path: Path) -> None:
    repaired, diagnostics = repair_track5_vertical_spikes(_submission_rows())
    paths = write_track5_vertical_repair_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        input_submission_path=tmp_path / "input.csv",
        template=_template_rows(),
        require_leaderboard_ready=True,
    )

    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["repaired_row_count"] == 1
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["results_csv"])
    assert official["Classification"].tolist() == [2, 2, 2, 2, 2]


def test_vertical_repair_cli_writes_outputs(tmp_path: Path) -> None:
    submission_csv = tmp_path / "submission.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    submission = _submission_rows()
    official = pd.DataFrame(
        {
            "Sequence": submission["sequence_id"],
            "Timestamp": submission["time_s"],
            "Position": [
                f"({row.state_x_m},{row.state_y_m},{row.state_z_m})"
                for row in submission.itertuples(index=False)
            ],
            "Classification": submission["Classification"],
        }
    )
    official.to_csv(submission_csv, index=False)
    _template_rows().to_csv(template_csv, index=False)

    status = vertical_repair_main(
        [
            "--submission",
            str(submission_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission_vertical_repair.zip").exists()
    assert (output_dir / "mmuad_track5_vertical_repair_manifest.json").exists()


def test_vertical_repair_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-vertical-repair"]
        == "raft_uav.mmuad.track5_vertical_repair:main"
    )
