from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd

from raft_uav.mmuad.track5_classification_relabel import main as relabel_main
from raft_uav.mmuad.track5_classification_relabel import relabel_track5_classification


def _pose_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,1)", "(1,0,1)", "(5,0,2)"],
            "Classification": [0, 0, 3],
        }
    )


def _classification_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(9,9,9)", "(8,8,8)", "(7,7,7)"],
            "Classification": [1, 1, 2],
        }
    )


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,0)", "(0,0,0)", "(0,0,0)"],
            "Classification": [1, 1, 2],
        }
    )


def test_track5_classification_relabel_copies_classes_and_preserves_positions() -> None:
    result = relabel_track5_classification(_pose_rows(), _classification_rows())

    assert result.rows["Position"].tolist() == ["(0,0,1)", "(1,0,1)", "(5,0,2)"]
    assert result.rows["Classification"].tolist() == [1, 1, 2]
    assert result.manifest["changed_row_count"] == 3
    assert result.diagnostics["classification_changed"].tolist() == [True, True, True]


def test_track5_classification_relabel_sequence_majority_mode() -> None:
    source = _classification_rows()
    source.loc[1, "Classification"] = 2

    result = relabel_track5_classification(_pose_rows(), source, mode="by-sequence-majority")

    seq1_labels = result.rows.loc[
        result.rows["Sequence"] == "seq0001",
        "Classification",
    ].tolist()
    assert seq1_labels == [1, 1]
    assert result.rows.loc[result.rows["Sequence"] == "seq0002", "Classification"].tolist() == [2]


def test_track5_classification_relabel_cli_writes_zip_and_validation(tmp_path: Path) -> None:
    pose_csv = tmp_path / "pose.csv"
    class_csv = tmp_path / "class.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _pose_rows().to_csv(pose_csv, index=False)
    _classification_rows().to_csv(class_csv, index=False)
    _template_rows().to_csv(template_csv, index=False)

    status = relabel_main(
        [
            "--pose-submission",
            str(pose_csv),
            "--classification-submission",
            str(class_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    results_csv = output_dir / "mmaud_results_relabelled.csv"
    zip_path = output_dir / "ug2_submission_relabelled.zip"
    manifest_path = output_dir / "mmuad_track5_classification_relabel_manifest.json"
    assert results_csv.exists()
    assert zip_path.exists()
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    results = pd.read_csv(results_csv)
    assert results["Position"].tolist() == ["(0,0,1)", "(1,0,1)", "(5,0,2)"]
    assert results["Classification"].tolist() == [1, 1, 2]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation"]["leaderboard_ready"] is True


def test_track5_classification_relabel_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-classification-relabel"]
        == "raft_uav.mmuad.track5_classification_relabel:main"
    )
