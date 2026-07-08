from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_hampel_repair import main as hampel_main
from raft_uav.mmuad.track5_hampel_repair import repair_track5_hampel_spikes
from raft_uav.mmuad.track5_hampel_repair import write_track5_hampel_repair_outputs


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 100.0, 3.0, 4.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "state_z_m": [5.0, 5.0, 50.0, 5.0, 5.0],
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


def test_hampel_repair_replaces_isolated_local_outlier() -> None:
    repaired, diagnostics = repair_track5_hampel_spikes(
        _submission_rows(),
        window_radius=2,
        sigma_threshold=2.0,
        min_scale_m=1.0,
        min_residual_m=5.0,
        repair_blend=1.0,
    )

    middle = repaired.loc[repaired["time_s"] == 2.0].iloc[0]
    assert middle["state_x_m"] == pytest.approx(2.0)
    assert middle["state_y_m"] == pytest.approx(0.0)
    assert middle["state_z_m"] == pytest.approx(5.0)
    assert bool(middle["hampel_repair_applied"])
    assert int(diagnostics["hampel_repair_applied"].sum()) == 1


def test_hampel_repair_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    repaired, diagnostics = repair_track5_hampel_spikes(
        _submission_rows(),
        window_radius=2,
        sigma_threshold=2.0,
        min_scale_m=1.0,
        min_residual_m=5.0,
    )
    paths = write_track5_hampel_repair_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
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
    assert manifest["changed_row_count"] == 1
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["results_csv"])
    assert official["Classification"].tolist() == [2, 2, 2, 2, 2]


def test_hampel_repair_cli_writes_outputs(tmp_path: Path) -> None:
    submission_csv = tmp_path / "submission.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _submission_rows().to_csv(submission_csv, index=False)
    _template_rows().to_csv(template_csv, index=False)

    status = hampel_main(
        [
            "--submission",
            str(submission_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--window-radius",
            "2",
            "--sigma-threshold",
            "2.0",
            "--min-residual-m",
            "5.0",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission_hampel_repaired.zip").exists()
    assert (output_dir / "mmuad_track5_hampel_repair_manifest.json").exists()


def test_hampel_repair_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-hampel-repair"]
        == "raft_uav.mmuad.track5_hampel_repair:main"
    )
