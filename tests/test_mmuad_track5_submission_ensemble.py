from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import ensemble_track5_submissions
from raft_uav.mmuad.track5_submission_ensemble import main as ensemble_main
from raft_uav.mmuad.track5_submission_ensemble import parse_submission_input
from raft_uav.mmuad.track5_submission_ensemble import write_track5_submission_ensemble_outputs


def _submission_rows(offset: float = 0.0, classification: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": [
                f"({0.0 + offset}, 0.0, 1.0)",
                f"({2.0 + offset}, 0.0, 1.0)",
                f"({10.0 + offset}, 1.0, 2.0)",
            ],
            "Classification": [classification, classification, 2],
        }
    )


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0, 0, 0)"] * 3,
            "Classification": [1, 1, 2],
        }
    )


def test_parse_submission_input_accepts_label_weight_path() -> None:
    parsed = parse_submission_input("smooth=0.25:/tmp/a.zip")

    assert parsed.label == "smooth"
    assert parsed.weight == pytest.approx(0.25)
    assert parsed.path.as_posix() == "/tmp/a.zip"


def test_track5_submission_ensemble_averages_positions_and_votes_classes(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(first, index=False)
    _submission_rows(offset=2.0, classification=3).to_csv(second, index=False)

    estimates, diagnostics = ensemble_track5_submissions(
        [parse_submission_input(f"a=1:{first}"), parse_submission_input(f"b=3:{second}")],
    )

    row = estimates.loc[(estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 0.0)].iloc[0]
    assert row["state_x_m"] == pytest.approx(1.5)
    assert row["Classification"] == 3
    assert diagnostics.loc[0, "input_count"] == 2
    assert diagnostics.loc[0, "position_spread_m"] > 0.0


def test_track5_submission_ensemble_rejects_template_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _submission_rows().to_csv(first, index=False)
    rows = _submission_rows()
    rows.loc[1, "Timestamp"] = 2.0
    rows.to_csv(second, index=False)

    with pytest.raises(ValueError, match="does not match the reference template keys"):
        ensemble_track5_submissions(
            [parse_submission_input(f"a={first}"), parse_submission_input(f"b={second}")],
        )


def test_track5_submission_ensemble_writes_zip_and_validation(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(first, index=False)
    _submission_rows(offset=2.0, classification=1).to_csv(second, index=False)
    estimates, diagnostics = ensemble_track5_submissions(
        [parse_submission_input(f"a={first}"), parse_submission_input(f"b={second}")],
    )

    paths = write_track5_submission_ensemble_outputs(
        estimates=estimates,
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        template=_template_rows(),
    )

    assert paths["zip"].exists()
    assert paths["results_csv"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_track5_submission_ensemble_cli_writes_manifest(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    template = tmp_path / "template.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(first, index=False)
    _submission_rows(offset=2.0, classification=1).to_csv(second, index=False)
    _template_rows().to_csv(template, index=False)
    output_dir = tmp_path / "out"

    status = ensemble_main(
        [
            "--submission",
            f"a=1:{first}",
            "--submission",
            f"b=1:{second}",
            "--template",
            str(template),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    manifest = json.loads((output_dir / "mmuad_track5_ensemble_manifest.json").read_text())
    assert manifest["row_count"] == 3
    assert manifest["validation"]["leaderboard_ready"] is True


def test_track5_submission_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-ensemble-track5-submissions"]
        == "raft_uav.mmuad.track5_submission_ensemble:main"
    )
