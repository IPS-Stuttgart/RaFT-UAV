from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_regularizer import main as regularizer_main
from raft_uav.mmuad.track5_trajectory_regularizer import regularize_track5_estimates
from raft_uav.mmuad.track5_trajectory_regularizer import run_track5_trajectory_regularizer


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 5,
            "Timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Position": ["(0,0,0)"] * 5,
            "Classification": [2] * 5,
        }
    )


def _outlier_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 100.0, 3.0, 4.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 1.0, 1.0, 1.0],
            "classification": [2, 2, 2, 2, 2],
        }
    )


def test_regularize_track5_estimates_suppresses_single_frame_outlier() -> None:
    regularized, diagnostics = regularize_track5_estimates(
        _outlier_estimates(),
        smoothness_weight=100.0,
        huber_delta_m=10.0,
        iterations=6,
        observation_sigma_m=1.0,
    )

    middle = regularized.loc[regularized["time_s"] == 2.0].iloc[0]
    assert middle["state_x_m"] < 50.0
    assert middle["regularizer_input_x_m"] == pytest.approx(100.0)
    assert diagnostics.loc[0, "finite_input_count"] == 5
    assert diagnostics.loc[0, "mean_robust_weight"] < 1.0


def test_track5_trajectory_regularizer_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    paths = run_track5_trajectory_regularizer(
        estimates=_outlier_estimates(),
        template=_template(),
        output_dir=tmp_path,
        class_map={"seq0001": "2"},
        smoothness_weight=100.0,
        huber_delta_m=10.0,
        iterations=6,
        observation_sigma_m=1.0,
    )

    assert paths["official_zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["regularized_rows"] == 5
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 2, 2, 2]


def test_track5_trajectory_regularizer_cli_writes_outputs(tmp_path: Path) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _outlier_estimates().to_csv(estimates_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001"], "uav_type": [2]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = regularizer_main(
        [
            "--estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--smoothness-weight",
            "100",
            "--huber-delta-m",
            "10",
            "--iterations",
            "6",
            "--observation-sigma-m",
            "1",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_regularizer_manifest.json").exists()


def test_track5_trajectory_regularizer_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-trajectory-regularizer"]
        == "raft_uav.mmuad.track5_trajectory_regularizer:main"
    )
