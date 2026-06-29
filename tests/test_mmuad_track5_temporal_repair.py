from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission
from raft_uav.mmuad.track5_temporal_repair import main as temporal_repair_main
from raft_uav.mmuad.track5_temporal_repair import repair_track5_temporal_spikes
from raft_uav.mmuad.track5_temporal_repair import write_track5_temporal_repair_outputs


def _official_submission_with_spike() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0, 3.0],
            "Position": ["(0, 0, 0)", "(100, 0, 0)", "(2, 0, 0)", "(3, 0, 0)"],
            "Classification": [2, 2, 2, 2],
        }
    )


def _smooth_official_submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0, 3.0],
            "Position": ["(0, 0, 0)", "(1, 0, 0)", "(2, 0, 0)", "(3, 0, 0)"],
            "Classification": [2, 2, 2, 2],
        }
    )


def test_temporal_repair_replaces_isolated_speed_spike(tmp_path: Path) -> None:
    submission = load_track5_submission(
        _write_frame_to_tmp(_official_submission_with_spike(), tmp_path)
    )
    repaired, diagnostics = repair_track5_temporal_spikes(
        submission,
        max_speed_mps=20.0,
        max_interpolation_residual_m=10.0,
        iterations=2,
    )

    middle = repaired.loc[repaired["time_s"] == 1.0].iloc[0]
    assert middle["state_x_m"] == pytest.approx(1.0)
    assert middle["Classification"] == 2
    assert diagnostics["repaired"].sum() == 1
    repaired_row = diagnostics.loc[diagnostics["repaired"]].iloc[0]
    assert repaired_row["repair_displacement_m"] == pytest.approx(99.0)


def test_temporal_repair_leaves_smooth_trajectory_unchanged(tmp_path: Path) -> None:
    submission = load_track5_submission(_write_frame_to_tmp(_smooth_official_submission(), tmp_path))
    repaired, diagnostics = repair_track5_temporal_spikes(
        submission,
        max_speed_mps=20.0,
        max_interpolation_residual_m=10.0,
    )

    assert repaired["state_x_m"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert diagnostics["repaired"].sum() == 0


def test_temporal_repair_outputs_upload_ready_zip(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    _official_submission_with_spike().to_csv(submission_path, index=False)
    submission = load_track5_submission(submission_path)
    repaired, diagnostics = repair_track5_temporal_spikes(
        submission,
        max_speed_mps=20.0,
        max_interpolation_residual_m=10.0,
    )

    paths = write_track5_temporal_repair_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        input_submission_path=submission_path,
        template=_official_submission_with_spike(),
        require_leaderboard_ready=True,
    )

    assert paths["zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert manifest["repaired_row_count"] == 1
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_temporal_repair_cli_writes_artifacts(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    output_dir = tmp_path / "out"
    _official_submission_with_spike().to_csv(submission_path, index=False)

    status = temporal_repair_main(
        [
            "--submission",
            str(submission_path),
            "--template",
            str(submission_path),
            "--output-dir",
            str(output_dir),
            "--max-speed-mps",
            "20",
            "--max-interpolation-residual-m",
            "10",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission_temporal_repair.zip").exists()
    assert (output_dir / "mmuad_track5_temporal_repair_manifest.json").exists()


def test_temporal_repair_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-temporal-repair"]
        == "raft_uav.mmuad.track5_temporal_repair:main"
    )


def _write_frame_to_tmp(frame: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "track5_temporal_repair_input.csv"
    frame.to_csv(path, index=False)
    return path
