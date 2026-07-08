from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_cv_smoother import main as cv_smoother_main
from raft_uav.mmuad.track5_cv_smoother import smooth_track5_cv_submission
from raft_uav.mmuad.track5_cv_smoother import write_track5_cv_smoother_outputs


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 10.0, 3.0, 4.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 1.0, 1.0, 1.0],
            "Classification": [2, 2, 2, 2, 2],
        }
    )


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 5,
            "Timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Position": ["(0,0,0)"] * 5,
            "Classification": [2] * 5,
        }
    )


def test_cv_smoother_reduces_isolated_constant_velocity_outlier() -> None:
    smoothed, diagnostics = smooth_track5_cv_submission(
        _submission_rows(),
        measurement_std_m=4.0,
        acceleration_std_mps2=0.3,
        initial_position_std_m=2.0,
        initial_velocity_std_mps=2.0,
        blend=1.0,
        max_correction_m=20.0,
    )

    raw_mid = _submission_rows().loc[2, "state_x_m"]
    smooth_mid = smoothed.loc[2, "state_x_m"]
    assert abs(smooth_mid - 2.0) < abs(raw_mid - 2.0)
    assert smoothed["Classification"].tolist() == [2, 2, 2, 2, 2]
    assert diagnostics.loc[2, "cv_smoother_applied_correction_m"] > 0.0


def test_cv_smoother_correction_cap_limits_large_motion() -> None:
    smoothed, diagnostics = smooth_track5_cv_submission(
        _submission_rows(),
        measurement_std_m=20.0,
        acceleration_std_mps2=0.1,
        blend=1.0,
        max_correction_m=1.0,
    )

    assert diagnostics["cv_smoother_capped"].any()
    applied = smoothed["cv_smoother_applied_correction_m"]
    assert applied.max() <= pytest.approx(1.0)


def test_cv_smoother_writes_leaderboard_ready_outputs(tmp_path: Path) -> None:
    smoothed, diagnostics = smooth_track5_cv_submission(_submission_rows())
    paths = write_track5_cv_smoother_outputs(
        smoothed=smoothed,
        diagnostics=diagnostics,
        output_dir=tmp_path,
        input_submission_path=tmp_path / "input.csv",
        template=_template_rows(),
        require_leaderboard_ready=True,
    )

    assert paths["zip"].exists()
    assert paths["results_csv"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_cv_smoother_cli_writes_artifacts(tmp_path: Path) -> None:
    submission_csv = tmp_path / "submission.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _submission_rows().to_csv(submission_csv, index=False)
    _template_rows().to_csv(template_csv, index=False)

    status = cv_smoother_main(
        [
            "--submission",
            str(submission_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--measurement-std-m",
            "4",
            "--acceleration-std-mps2",
            "0.3",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission_cv_smoothed.zip").exists()
    assert (output_dir / "mmuad_track5_cv_smoother_manifest.json").exists()


def test_cv_smoother_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-cv-smoother"]
        == "raft_uav.mmuad.track5_cv_smoother:main"
    )
