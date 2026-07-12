from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_scorecard_cli import main as track5_scorecard_main



def _write_official_results_zip(path: Path) -> None:
    rows = pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(1.0,2.0,3.0)", "(2.0,2.0,3.0)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mmaud_results.csv", rows.to_csv(index=False))



def _write_public_sequence_root_zip(path: Path) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("val/seq001/Image/0.0.png", b"")
        archive.writestr("val/seq001/Image/1.0.png", b"")



def test_track5_scorecard_cli_accepts_archived_sequence_root(tmp_path: Path) -> None:
    """The one-stop scorecard can build its template from a zipped public split."""

    results_zip = tmp_path / "ug2_submission.zip"
    sequence_zip = tmp_path / "mmuad_val.zip"
    truth_csv = tmp_path / "truth.csv"
    output_json = tmp_path / "scorecard.json"
    summary_csv = tmp_path / "scorecard.csv"
    archive_manifest_json = tmp_path / "scorecard_sequence_root_archive_manifest.json"

    _write_official_results_zip(results_zip)
    _write_public_sequence_root_zip(sequence_zip)
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "x_m": [1.0, 2.0],
            "y_m": [2.0, 2.0],
            "z_m": [3.0, 3.0],
            "uav_type": ["2", "2"],
        }
    ).to_csv(truth_csv, index=False)

    rc = track5_scorecard_main(
        [
            "--results",
            str(results_zip),
            "--truth",
            str(truth_csv),
            "--sequence-root",
            str(sequence_zip),
            "--split-name",
            "val",
            "--timestamp-source",
            "image",
            "--sequence-root-archive-manifest-json",
            str(archive_manifest_json),
            "--output-json",
            str(output_json),
            "--summary-csv",
            str(summary_csv),
        ]
    )

    assert rc == 0
    assert output_json.is_file()
    assert summary_csv.is_file()
    assert archive_manifest_json.is_file()

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    manifest = json.loads(archive_manifest_json.read_text(encoding="utf-8"))

    assert summary["sequence_root_archive_manifest_json"] == str(archive_manifest_json)
    assert summary["validation"]["template_timestamp_count"] == 2
    assert summary["scorecard_leaderboard_ready"] is True
    assert summary["codabench_upload_ready"] is True
    assert manifest["archive_format"] == "zip"
    assert Path(manifest["extract_root"]).is_dir()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_track5_scorecard_cli_rejects_nonfinite_timestamp_tolerance(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="timestamp_tolerance_s must be finite and non-negative",
    ):
        track5_scorecard_main(
            [
                "--results",
                str(tmp_path / "missing.zip"),
                "--output-json",
                str(tmp_path / "scorecard.json"),
                "--timestamp-tolerance-s",
                value,
            ]
        )


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_track5_scorecard_cli_rejects_nonfinite_nearest_time_delta(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="nearest_time_delta_s must be finite and non-negative",
    ):
        track5_scorecard_main(
            [
                "--results",
                str(tmp_path / "missing.zip"),
                "--output-json",
                str(tmp_path / "scorecard.json"),
                "--nearest-time-delta-s",
                value,
            ]
        )


def test_track5_scorecard_cli_rejects_negative_time_gate(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="timestamp_tolerance_s must be finite and non-negative",
    ):
        track5_scorecard_main(
            [
                "--results",
                str(tmp_path / "missing.zip"),
                "--output-json",
                str(tmp_path / "scorecard.json"),
                "--timestamp-tolerance-s",
                "-1",
            ]
        )
