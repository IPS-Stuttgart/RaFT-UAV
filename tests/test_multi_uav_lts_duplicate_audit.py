from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.duplicate_audit import (
    audit_duplicate_predictions,
    main as duplicate_audit_main,
)


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def _row(frame: int, object_id: int) -> str:
    return f"{frame},{object_id},10,20,5,6,0.9,1,1\n"


def test_duplicate_prediction_audit_detects_repeated_frame_object_key(
    tmp_path: Path,
) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("S_00.txt").write_text(
        _row(1, 7) + _row(1, 7) + _row(2, 7),
        encoding="utf-8",
    )

    audit = audit_duplicate_predictions(prediction_dir)

    assert not audit.clean
    assert audit.file_count == 1
    assert audit.total_rows == 3
    assert audit.duplicate_key_count == 1
    assert audit.duplicate_rows == 1
    assert audit.duplicate_files == ["S_00.txt"]
    duplicate = audit.duplicate_keys[0]
    assert duplicate.name == "S_00.txt"
    assert duplicate.frame_id == 1
    assert duplicate.object_id == 7
    assert duplicate.occurrence_count == 2


def test_duplicate_prediction_audit_handles_zip_inputs(tmp_path: Path) -> None:
    submission = tmp_path / "submission.zip"
    _write_zip(
        submission,
        {
            "S_00.txt": _row(1, 1) + _row(1, 1),
            "S_01.txt": _row(1, 2),
        },
    )

    audit = audit_duplicate_predictions(submission)

    assert not audit.clean
    assert audit.file_count == 2
    assert audit.duplicate_files == ["S_00.txt"]
    assert audit.duplicate_rows == 1


def test_duplicate_prediction_audit_cli_writes_outputs(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    output_json = tmp_path / "duplicate_audit.json"
    file_summary_csv = tmp_path / "duplicate_file_summary.csv"
    duplicate_keys_csv = tmp_path / "duplicate_keys.csv"
    prediction_dir.mkdir()
    prediction_dir.joinpath("S_00.txt").write_text(
        _row(1, 5) + _row(1, 5),
        encoding="utf-8",
    )

    status = duplicate_audit_main(
        [
            str(prediction_dir),
            "--output-json",
            str(output_json),
            "--file-summary-csv",
            str(file_summary_csv),
            "--duplicate-keys-csv",
            str(duplicate_keys_csv),
        ]
    )

    assert status == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["clean"] is False
    assert payload["duplicate_rows"] == 1
    assert "S_00.txt" in file_summary_csv.read_text(encoding="utf-8")
    assert "frame_id" in duplicate_keys_csv.read_text(encoding="utf-8")


def test_duplicate_prediction_audit_require_clean_exits_nonzero(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("S_00.txt").write_text(
        _row(1, 5) + _row(1, 5),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        duplicate_audit_main([str(prediction_dir), "--require-clean"])

    assert exc_info.value.code == 1
