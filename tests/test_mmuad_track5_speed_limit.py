from __future__ import annotations

import json
import math
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import main as speed_limit_main
from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit
from raft_uav.mmuad.track5_speed_limit import write_track5_speed_limit_outputs


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 2.0, 0.0],
            "state_x_m": [0.0, 100.0, 200.0, 5.0],
            "state_y_m": [0.0, 0.0, 0.0, 5.0],
            "state_z_m": [0.0, 0.0, 0.0, 5.0],
            "Classification": [2, 2, 2, 1],
        }
    )


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 2.0, 0.0],
            "Position": ["(0,0,0)"] * 4,
            "Classification": [2, 2, 2, 1],
        }
    )


def test_speed_limit_projection_caps_consecutive_motion() -> None:
    limited, diagnostics = project_track5_speed_limit(
        _submission_rows(),
        max_speed_mps=10.0,
        iterations=2,
    )
    seq = limited.loc[limited["sequence_id"] == "seq0001"].sort_values("time_s")

    assert seq["state_x_m"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert seq["Classification"].tolist() == [2, 2, 2]
    assert diagnostics["speed_limit_applied"].sum() == 2
    assert diagnostics.loc[diagnostics["sequence_id"] == "seq0001", "output_speed_prev_mps"].max() <= 10.0


@pytest.mark.parametrize("max_speed_mps", [math.nan, math.inf])
def test_speed_limit_rejects_non_finite_max_speed(max_speed_mps: float) -> None:
    with pytest.raises(ValueError, match="max_speed_mps must be positive and finite"):
        project_track5_speed_limit(
            _submission_rows(),
            max_speed_mps=max_speed_mps,
            iterations=2,
        )


def test_speed_limit_outputs_write_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    limited, diagnostics = project_track5_speed_limit(
        _submission_rows(),
        max_speed_mps=10.0,
        iterations=2,
    )
    paths = write_track5_speed_limit_outputs(
        limited=limited,
        diagnostics=diagnostics,
        output_dir=tmp_path,
        input_submission_path=tmp_path / "input.csv",
        template=_template_rows(),
        require_leaderboard_ready=True,
    )

    assert paths["zip"].exists()
    assert paths["results_csv"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["changed_row_count"] == 2
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["results_csv"])
    assert official["Classification"].tolist() == [2, 2, 2, 1]


def test_speed_limit_cli_writes_outputs(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    template = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _submission_rows().to_csv(submission, index=False)
    _template_rows().to_csv(template, index=False)

    status = speed_limit_main(
        [
            "--submission",
            str(submission),
            "--template",
            str(template),
            "--output-dir",
            str(output_dir),
            "--max-speed-mps",
            "10",
            "--iterations",
            "2",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission_speed_limited.zip").exists()
    assert (output_dir / "mmuad_track5_speed_limit_manifest.json").exists()


def test_speed_limit_cli_preserves_normalized_csv_sequence_ids(tmp_path: Path) -> None:
    submission = tmp_path / "normalized_submission.csv"
    template = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["001", "001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "Classification": [2, 2],
        }
    ).to_csv(submission, index=False)
    pd.DataFrame(
        {
            "Sequence": ["001", "001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2],
        }
    ).to_csv(template, index=False)

    assert (
        speed_limit_main(
            [
                "--submission",
                str(submission),
                "--template",
                str(template),
                "--output-dir",
                str(output_dir),
                "--max-speed-mps",
                "10",
                "--require-leaderboard-ready",
            ]
        )
        == 0
    )

    official = pd.read_csv(
        output_dir / "mmaud_results_speed_limited.csv",
        dtype={"Sequence": "string"},
        keep_default_na=False,
    )
    validation = json.loads(
        (output_dir / "mmuad_track5_speed_limit_validation.json").read_text(encoding="utf-8")
    )
    assert official["Sequence"].tolist() == ["001", "001"]
    assert validation["leaderboard_ready"] is True


def test_speed_limit_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-speed-limit"]
        == "raft_uav.mmuad.track5_speed_limit:main"
    )
