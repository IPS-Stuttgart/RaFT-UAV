from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_medoid_ensemble import (
    ensemble_track5_submissions_medoid,
)
from raft_uav.mmuad.track5_submission_medoid_ensemble import main as medoid_main
from raft_uav.mmuad.track5_submission_medoid_ensemble import (
    write_track5_submission_medoid_ensemble_outputs,
)
from raft_uav.mmuad.track5_submission_ensemble import parse_submission_input


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


def test_track5_submission_medoid_ensemble_rejects_outlier_submission(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    outlier = tmp_path / "outlier.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(first, index=False)
    _submission_rows(offset=2.0, classification=3).to_csv(second, index=False)
    _submission_rows(offset=1000.0, classification=3).to_csv(outlier, index=False)

    estimates, diagnostics = ensemble_track5_submissions_medoid(
        [
            parse_submission_input(f"a=1:{first}"),
            parse_submission_input(f"b=1:{second}"),
            parse_submission_input(f"outlier=1:{outlier}"),
        ]
    )

    row = estimates.loc[
        (estimates["sequence_id"] == "seq0001") & (estimates["time_s"] == 0.0)
    ].iloc[0]
    assert row["state_x_m"] == pytest.approx(2.0)
    assert row["medoid_selected_label"] == "b"
    assert row["Classification"] == 3
    assert diagnostics.loc[0, "selected_label"] == "b"
    assert diagnostics.loc[0, "center_policy"] == "weighted-median"


def test_track5_submission_medoid_ensemble_writes_zip_and_validation(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    outlier = tmp_path / "outlier.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(first, index=False)
    _submission_rows(offset=2.0, classification=1).to_csv(second, index=False)
    _submission_rows(offset=1000.0, classification=1).to_csv(outlier, index=False)
    inputs = [
        parse_submission_input(f"a=1:{first}"),
        parse_submission_input(f"b=1:{second}"),
        parse_submission_input(f"outlier=1:{outlier}"),
    ]
    estimates, diagnostics = ensemble_track5_submissions_medoid(inputs)

    paths = write_track5_submission_medoid_ensemble_outputs(
        estimates=estimates,
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        inputs=inputs,
        center_policy="weighted-median",
        class_policy="weighted-vote",
        template=_template_rows(),
        require_leaderboard_ready=True,
    )

    assert paths["zip"].exists()
    assert paths["results_csv"].exists()
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["schema"] == "raft-uav-mmuad-track5-submission-ensemble-v1"
    assert manifest["center_policy"] == "weighted-median"
    assert manifest["validation"]["leaderboard_ready"] is True
    with ZipFile(paths["zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_track5_submission_medoid_ensemble_cli_writes_manifest(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    outlier = tmp_path / "outlier.csv"
    template = tmp_path / "template.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(first, index=False)
    _submission_rows(offset=2.0, classification=1).to_csv(second, index=False)
    _submission_rows(offset=1000.0, classification=1).to_csv(outlier, index=False)
    _template_rows().to_csv(template, index=False)
    output_dir = tmp_path / "out"

    status = medoid_main(
        [
            "--submission",
            f"a=1:{first}",
            "--submission",
            f"b=1:{second}",
            "--submission",
            f"outlier=1:{outlier}",
            "--template",
            str(template),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    manifest = json.loads((output_dir / "mmuad_track5_ensemble_manifest.json").read_text())
    assert manifest["center_policy"] == "weighted-median"
    assert manifest["validation"]["leaderboard_ready"] is True


def test_track5_submission_medoid_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-medoid-track5-submissions"]
        == "raft_uav.mmuad.track5_submission_medoid_ensemble:main"
    )
