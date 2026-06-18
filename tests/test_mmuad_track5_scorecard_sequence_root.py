from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_scorecard import (
    build_track5_scorecard,
    template_frame_from_sequence_root,
)
from raft_uav.mmuad.track5_scorecard_cli import main as track5_scorecard_cli_main


def _write_official_zip(path: Path, frame: pd.DataFrame) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))
    return path


def _make_public_sequence_root(tmp_path: Path) -> Path:
    sequence = tmp_path / "mmuad_public" / "val" / "seq0001" / "Image"
    sequence.mkdir(parents=True)
    # The adapter uses the numeric filename token as public Track 5 timestamp.
    (sequence / "1.000000.png").write_bytes(b"not-a-real-image-but-a-frame-placeholder")
    (sequence / "2.000000.png").write_bytes(b"not-a-real-image-but-a-frame-placeholder")
    return tmp_path / "mmuad_public"


def test_track5_scorecard_template_can_come_from_sequence_root(tmp_path: Path) -> None:
    sequence_root = _make_public_sequence_root(tmp_path)
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq0001", "seq0001"],
                "Timestamp": [1.0, 2.0],
                "Position": ["(0,0,10)", "(1,0,10)"],
                "Classification": [0, 0],
            }
        ),
    )

    template = template_frame_from_sequence_root(
        sequence_root,
        split_name="val",
        timestamp_source="image",
    )
    scorecard = build_track5_scorecard(
        results_path=results_zip,
        sequence_root=sequence_root,
        split_name="val",
        timestamp_source="image",
    )

    assert template[["sequence_id", "time_s"]].to_dict("records") == [
        {"sequence_id": "seq0001", "time_s": 1.0},
        {"sequence_id": "seq0001", "time_s": 2.0},
    ]
    validation = scorecard.summary["validation"]
    assert validation["template_checked"] is True
    assert validation["template_timestamp_count"] == 2
    assert validation["missing_template_timestamp_count"] == 0
    assert validation["extra_prediction_count"] == 0
    assert validation["codabench_upload_ready"] is True
    # Hidden validation/test labels are unavailable, so the overall scorecard is
    # intentionally not leaderboard-ready without --truth.
    assert scorecard.summary["scorecard_leaderboard_ready"] is False
    assert scorecard.summary["leaderboard_blocking_reasons"] == [
        "public_track5_evaluation_not_run"
    ]


def test_track5_scorecard_cli_sequence_root_template(tmp_path: Path, capsys) -> None:
    sequence_root = _make_public_sequence_root(tmp_path)
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq0001", "seq0001"],
                "Timestamp": [1.0, 2.0],
                "Position": ["(0,0,10)", "(1,0,10)"],
                "Classification": [0, 0],
            }
        ),
    )
    output_json = tmp_path / "scorecard.json"

    with pytest.raises(SystemExit, match="public_track5_evaluation_not_run"):
        track5_scorecard_cli_main(
            [
                "--results",
                str(results_zip),
                "--sequence-root",
                str(sequence_root),
                "--split-name",
                "val",
                "--timestamp-source",
                "image",
                "--output-json",
                str(output_json),
                "--require-leaderboard-ready",
            ]
        )

    stdout = capsys.readouterr().out
    assert "track5_scorecard=ok" in stdout
    assert "codabench_upload_ready=True" in stdout
    assert "template_timestamp_count=2" in stdout
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["sequence_root"] == str(sequence_root)
    assert summary["split_name"] == "val"
    assert summary["timestamp_source"] == "image"
    assert summary["validation"]["codabench_upload_ready"] is True
