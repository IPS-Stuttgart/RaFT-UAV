from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_jerk_limit import main as jerk_limit_main
from raft_uav.mmuad.track5_jerk_limit import repair_track5_jerk_kinks
from raft_uav.mmuad.track5_jerk_limit import write_track5_jerk_limit_outputs


def _normalized_submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_x_m": [0.0, 1.0, 30.0, 3.0, 4.0, 5.0],
            "state_y_m": [0.0] * 6,
            "state_z_m": [1.0] * 6,
            "Classification": [2] * 6,
        }
    )


def _official_submission() -> pd.DataFrame:
    rows = _normalized_submission()
    return pd.DataFrame(
        {
            "Sequence": rows["sequence_id"],
            "Timestamp": rows["time_s"],
            "Position": [
                f"({x},{y},{z})"
                for x, y, z in zip(
                    rows["state_x_m"],
                    rows["state_y_m"],
                    rows["state_z_m"],
                    strict=False,
                )
            ],
            "Classification": rows["Classification"],
        }
    )


def test_jerk_limit_repairs_short_oscillatory_kink() -> None:
    repaired, diagnostics = repair_track5_jerk_kinks(
        _normalized_submission(),
        max_jerk_mps3=5.0,
        smoothness_weight=100.0,
        min_correction_m=1.0,
        iterations=1,
    )

    spike = repaired.loc[repaired["time_s"] == 2.0, "state_x_m"].iloc[0]
    assert spike < 15.0
    assert diagnostics.loc[diagnostics["time_s"] == 2.0, "jerk_limit_applied"].iloc[0]
    assert repaired["Classification"].tolist() == [2] * 6


def test_jerk_limit_diagnostics_report_actual_blended_displacement() -> None:
    original = _normalized_submission()
    repaired, diagnostics = repair_track5_jerk_kinks(
        original,
        max_jerk_mps3=5.0,
        smoothness_weight=100.0,
        min_correction_m=1.0,
        iterations=1,
        repair_blend=0.5,
    )

    coordinate_columns = ["state_x_m", "state_y_m", "state_z_m"]
    actual_displacement = np.linalg.norm(
        repaired[coordinate_columns].to_numpy(float)
        - original[coordinate_columns].to_numpy(float),
        axis=1,
    )
    reported_displacement = diagnostics["jerk_limit_displacement_m"].to_numpy(float)

    assert diagnostics["jerk_limit_applied"].any()
    assert np.allclose(reported_displacement, actual_displacement)


def test_jerk_limit_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    repaired, diagnostics = repair_track5_jerk_kinks(
        _official_submission(),
        max_jerk_mps3=5.0,
        smoothness_weight=100.0,
        min_correction_m=1.0,
    )
    paths = write_track5_jerk_limit_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=tmp_path,
        input_submission_path=tmp_path / "input.csv",
        template=_official_submission(),
        require_leaderboard_ready=True,
    )

    assert paths["zip"].exists()
    assert paths["results_csv"].exists()
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    assert manifest["changed_row_count"] >= 1
    assert validation["leaderboard_ready"] is True
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_jerk_limit_cli_writes_outputs(tmp_path: Path) -> None:
    submission_csv = tmp_path / "mmaud_results.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _official_submission().to_csv(submission_csv, index=False)
    _official_submission().to_csv(template_csv, index=False)

    status = jerk_limit_main(
        [
            "--submission",
            str(submission_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--max-jerk-mps3",
            "5",
            "--smoothness-weight",
            "100",
            "--min-correction-m",
            "1",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission_jerk_limited.zip").exists()
    assert (output_dir / "mmuad_track5_jerk_limit_manifest.json").exists()


def test_jerk_limit_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-jerk-limit"]
        == "raft_uav.mmuad.track5_jerk_limit:main"
    )
