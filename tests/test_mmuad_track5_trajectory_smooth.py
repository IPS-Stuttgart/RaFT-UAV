from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.submission import write_official_mmaud_results_csv
from raft_uav.mmuad.track5_trajectory_smooth import main as smooth_main
from raft_uav.mmuad.track5_trajectory_smooth import smooth_track5_submission_rows
from raft_uav.mmuad.track5_trajectory_smooth import write_track5_trajectory_smooth_outputs


def _normalized_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 12.0, 3.0, 4.0],
            "state_y_m": [0.0, 0.0, 10.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 11.0, 1.0, 1.0],
            "Classification": [2] * 5,
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 5,
            "Timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Position": ["(0,0,0)"] * 5,
            "Classification": [2] * 5,
        }
    )


def test_trajectory_smoothing_reduces_isolated_spike() -> None:
    smoothed, diagnostics = smooth_track5_submission_rows(
        _normalized_rows(),
        window_s=2.0,
        bandwidth_s=1.0,
        blend=1.0,
        max_correction_m=None,
        min_neighbors=3,
    )

    middle = smoothed.loc[smoothed["time_s"] == 2.0].iloc[0]
    original_middle = _normalized_rows().loc[_normalized_rows()["time_s"] == 2.0].iloc[0]
    assert middle["state_x_m"] < original_middle["state_x_m"]
    assert middle["state_y_m"] < original_middle["state_y_m"]
    assert middle["Classification"] == 2
    assert diagnostics.loc[diagnostics["time_s"] == 2.0, "neighbor_count"].iloc[0] == 5


def test_trajectory_smoothing_preserves_constant_velocity() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_y_m": [1.0, 1.0, 1.0, 1.0, 1.0],
            "state_z_m": [2.0, 2.0, 2.0, 2.0, 2.0],
            "Classification": [1] * 5,
        }
    )
    smoothed, _ = smooth_track5_submission_rows(
        rows,
        window_s=2.0,
        bandwidth_s=1.0,
        blend=1.0,
        max_correction_m=None,
        min_neighbors=3,
    )

    assert smoothed["state_x_m"].tolist() == pytest.approx(rows["state_x_m"].tolist())
    assert smoothed["state_y_m"].tolist() == pytest.approx(rows["state_y_m"].tolist())
    assert smoothed["state_z_m"].tolist() == pytest.approx(rows["state_z_m"].tolist())


def test_trajectory_smoothing_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    paths = write_track5_trajectory_smooth_outputs(
        rows=_normalized_rows(),
        output_dir=tmp_path,
        template=_template(),
        window_s=2.0,
        bandwidth_s=1.0,
        blend=0.5,
        max_correction_m=5.0,
    )

    assert paths["zip"].exists()
    assert paths["results_csv"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["row_count"] == 5
    assert manifest["max_correction_m"] == 5.0
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_trajectory_smoothing_cli_writes_outputs(tmp_path: Path) -> None:
    submission_csv = tmp_path / "mmaud_results.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    write_official_mmaud_results_csv(
        _normalized_rows(),
        submission_csv,
        classification=2,
        invalid_row_policy="raise",
    )
    _template().to_csv(template_csv, index=False)

    status = smooth_main(
        [
            "--submission",
            str(submission_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--window-s",
            "2",
            "--bandwidth-s",
            "1",
            "--blend",
            "0.5",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission_smoothed.zip").exists()
    assert (output_dir / "mmuad_track5_trajectory_smooth_manifest.json").exists()


def test_trajectory_smoothing_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-trajectory-smooth"]
        == "raft_uav.mmuad.track5_trajectory_smooth:main"
    )
