from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.cli import (
    inventory_path,
    normalize_prediction_text,
    package_submission,
    validate_submission_zip,
    write_constant_first_frame_predictions,
)


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def test_validate_submission_zip_against_template(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    submission = tmp_path / "submission.zip"
    row = "1,1,10.0,20.0,5.0,6.0,0.9,1,1.0\n"
    _write_zip(template, {"A_00.txt": "", "B_00.txt": ""})
    _write_zip(submission, {"A_00.txt": row, "B_00.txt": row + "2,1,11,20,5,6,0.8,1,1\n"})

    validation = validate_submission_zip(submission, template_zip=template)

    assert validation.valid
    assert validation.file_count == 2
    assert validation.expected_file_count == 2
    assert validation.total_rows == 3
    assert validation.files[0].first_frame == 1
    assert validation.files[1].last_frame == 2


def test_validate_submission_rejects_nested_and_bad_rows(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    submission = tmp_path / "submission.zip"
    _write_zip(template, {"A_00.txt": ""})
    _write_zip(
        submission,
        {
            "nested/A_00.txt": "1,1,10,20,5,6,1,1,1\n",
            "A_00.txt": "1,1,10,20,-5,6,1.2,1,1\n",
            "notes.md": "nope\n",
        },
    )

    validation = validate_submission_zip(submission, template_zip=template)

    assert not validation.valid
    assert validation.nested_entries == ["nested/A_00.txt"]
    assert validation.non_txt_entries == ["notes.md"]
    assert validation.invalid_geometry_rows == 1
    assert validation.invalid_confidence_rows == 1


def test_package_submission_fills_missing_template_files(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    output_zip = tmp_path / "submission.zip"
    prediction_dir.mkdir()
    _write_zip(template, {"A_00.txt": "", "B_00.txt": ""})
    (prediction_dir / "A_00.txt").write_text("1,1,10,20,5,6,1,1,1\n", encoding="utf-8")

    validation = package_submission(prediction_dir, output_zip, template_zip=template)

    assert validation.valid
    assert validation.file_count == 2
    with zipfile.ZipFile(output_zip) as archive:
        assert sorted(archive.namelist()) == ["A_00.txt", "B_00.txt"]
        assert archive.read("B_00.txt") == b""


def test_package_submission_can_normalize_float_ids(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    output_zip = tmp_path / "submission.zip"
    prediction_dir.mkdir()
    _write_zip(template, {"A_00.txt": ""})
    (prediction_dir / "A_00.txt").write_text(
        "1.000000,2.000000,10.500000,20.000000,5.000000,6.000000,-1.000000,1.000000,1.000000\n",
        encoding="utf-8",
    )

    validation = package_submission(prediction_dir, output_zip, template_zip=template, normalize=True)

    assert validation.valid
    with zipfile.ZipFile(output_zip) as archive:
        assert archive.read("A_00.txt").decode() == "1,2,10.5,20,5,6,-1,1,1\n"


def test_package_submission_can_sort_rows(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    output_zip = tmp_path / "submission.zip"
    prediction_dir.mkdir()
    _write_zip(template, {"A_00.txt": ""})
    (prediction_dir / "A_00.txt").write_text(
        "2,2,10,20,5,6,1,1,1\n"
        "1,3,10,20,5,6,1,1,1\n"
        "1,1,10,20,5,6,1,1,1\n",
        encoding="utf-8",
    )

    validation = package_submission(prediction_dir, output_zip, template_zip=template, sort_rows=True)

    assert validation.valid
    assert validation.unsorted_rows == 0
    with zipfile.ZipFile(output_zip) as archive:
        assert archive.read("A_00.txt").decode().splitlines() == [
            "1,1,10,20,5,6,1,1,1",
            "1,3,10,20,5,6,1,1,1",
            "2,2,10,20,5,6,1,1,1",
        ]


def test_normalize_prediction_text_rejects_non_integer_ids() -> None:
    with pytest.raises(ValueError, match="integer-like"):
        normalize_prediction_text("1.2,2,10,20,5,6,1,1,1\n")


def test_inventory_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "Train.zip"
    _write_zip(archive_path, {"Train/A_00.mp4": "video", "Train/A_00.txt": "labels"})

    summary = inventory_path(archive_path)

    assert summary["source_type"] == "zip"
    assert summary["file_count"] == 2
    assert summary["suffix_counts"][".mp4"] == 1
    assert summary["top_level_counts"]["Train"] == 2
    json.dumps(summary)


def test_constant_first_frame_predictions(tmp_path: Path) -> None:
    sequence_root = tmp_path / "TestImages"
    labels = tmp_path / "TestLabels_FirstFrameOnly"
    predictions = tmp_path / "predictions"
    (sequence_root / "S_00").mkdir(parents=True)
    labels.mkdir()
    for frame in ["00000.jpg", "00001.jpg", "00002.jpg"]:
        (sequence_root / "S_00" / frame).write_bytes(b"fake")
    labels.joinpath("S_00.txt").write_text(
        "1,7,10.5,20.0,5.0,6.0,1,1,1.0\n"
        "1,8,30.0,40.0,7.0,8.0,0.5,1,0.8\n",
        encoding="utf-8",
    )

    summary = write_constant_first_frame_predictions(sequence_root, labels, predictions)

    assert summary["sequence_count"] == 1
    assert summary["total_rows"] == 6
    lines = (predictions / "S_00.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "1,7,10.5,20,5,6,1,1,1"
    assert lines[-1] == "3,8,30,40,7,8,0.5,1,0.8"
