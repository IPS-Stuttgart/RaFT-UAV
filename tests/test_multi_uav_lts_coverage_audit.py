from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.coverage_audit import (
    audit_prediction_coverage,
    main as coverage_audit_main,
)


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def _row(
    frame: int = 1,
    object_id: int = 1,
    *,
    class_id: int = 1,
    visibility: float = 1.0,
) -> str:
    return f"{frame},{object_id},10,20,5,6,0.9,{class_id},{visibility}\n"


def _sequence_root(tmp_path: Path, names: tuple[str, ...] = ("A_00",)) -> Path:
    root = tmp_path / "TestImages"
    for name in names:
        sequence_dir = root / name
        sequence_dir.mkdir(parents=True)
        for frame in range(1, 4):
            sequence_dir.joinpath(f"{frame:06d}.jpg").write_text("", encoding="utf-8")
    return root


def test_prediction_coverage_audit_detects_missing_extra_and_empty_files(tmp_path: Path) -> None:
    template_zip = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_zip(template_zip, {"A_00.txt": "", "B_00.txt": "", "C_00.txt": ""})
    prediction_dir.joinpath("A_00.txt").write_text(_row(), encoding="utf-8")
    prediction_dir.joinpath("B_00.txt").write_text("", encoding="utf-8")
    prediction_dir.joinpath("EXTRA_00.txt").write_text(_row(), encoding="utf-8")

    audit = audit_prediction_coverage(prediction_dir, template_zip=template_zip)
    rows_by_name = {row.name: row for row in audit.rows}

    assert not audit.ready
    assert audit.expected_file_count == 3
    assert audit.present_file_count == 3
    assert audit.missing_files == ["C_00.txt"]
    assert audit.extra_files == ["EXTRA_00.txt"]
    assert audit.empty_expected_files == ["B_00.txt"]
    assert audit.blocking_reasons == ["missing_files", "extra_files", "empty_expected_files"]
    assert rows_by_name["A_00.txt"].status == "ok"
    assert rows_by_name["B_00.txt"].status == "empty_expected"
    assert rows_by_name["C_00.txt"].status == "missing"
    assert rows_by_name["EXTRA_00.txt"].status == "extra"


def test_prediction_coverage_audit_empty_expected_file_is_not_ready(tmp_path: Path) -> None:
    template_zip = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_zip(template_zip, {"A_00.txt": ""})
    prediction_dir.joinpath("A_00.txt").write_text("", encoding="utf-8")

    audit = audit_prediction_coverage(prediction_dir, template_zip=template_zip)

    assert not audit.ready
    assert audit.missing_files == []
    assert audit.extra_files == []
    assert audit.empty_expected_file_count == 1
    assert audit.empty_expected_files == ["A_00.txt"]
    assert audit.blocking_reasons == ["empty_expected_files"]
    assert audit.rows[0].status == "empty_expected"

    with pytest.raises(SystemExit) as exc_info:
        coverage_audit_main(
            [
                str(prediction_dir),
                "--template-zip",
                str(template_zip),
                "--require-ready",
            ]
        )

    assert exc_info.value.code == 1


def test_prediction_coverage_audit_uses_sequence_root_names(tmp_path: Path) -> None:
    sequence_root = tmp_path / "TestImages"
    prediction_dir = tmp_path / "predictions"
    (sequence_root / "S_00").mkdir(parents=True)
    (sequence_root / "S_01").mkdir(parents=True)
    prediction_dir.mkdir()
    prediction_dir.joinpath("S_00.txt").write_text(_row(), encoding="utf-8")

    audit = audit_prediction_coverage(prediction_dir, sequence_root=sequence_root)

    assert not audit.ready
    assert audit.expected_file_count == 2
    assert audit.missing_files == ["S_01.txt"]
    assert audit.blocking_reasons == ["missing_files"]


def test_prediction_coverage_audit_detects_frame_ids_beyond_sequence_length(
    tmp_path: Path,
) -> None:
    sequence_root = _sequence_root(tmp_path, names=("A_00",))
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("A_00.txt").write_text(
        _row(frame=1) + _row(frame=4, object_id=2),
        encoding="utf-8",
    )

    audit = audit_prediction_coverage(prediction_dir, sequence_root=sequence_root)
    row = audit.rows[0]

    assert not audit.ready
    assert audit.out_of_range_frame_rows == 1
    assert audit.out_of_range_frame_files == ["A_00.txt"]
    assert audit.blocking_reasons == ["out_of_range_frame_rows"]
    assert row.status == "invalid"
    assert row.expected_frame_count == 3
    assert row.out_of_range_frame_rows == 1


def test_prediction_coverage_audit_ignores_nested_images_for_sequence_length(
    tmp_path: Path,
) -> None:
    sequence_root = _sequence_root(tmp_path, names=("A_00",))
    nested_dir = sequence_root / "A_00" / "diagnostics"
    nested_dir.mkdir()
    nested_dir.joinpath("debug_overlay.jpg").write_text("", encoding="utf-8")
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("A_00.txt").write_text(_row(frame=4), encoding="utf-8")

    audit = audit_prediction_coverage(prediction_dir, sequence_root=sequence_root)
    row = audit.rows[0]

    assert not audit.ready
    assert row.expected_frame_count == 3
    assert audit.out_of_range_frame_rows == 1
    assert audit.out_of_range_frame_files == ["A_00.txt"]
    assert audit.blocking_reasons == ["out_of_range_frame_rows"]
    assert row.status == "invalid"


def test_prediction_coverage_audit_detects_duplicate_frame_object_rows(tmp_path: Path) -> None:
    template_zip = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_zip(template_zip, {"A_00.txt": ""})
    prediction_dir.joinpath("A_00.txt").write_text(
        _row(frame=1, object_id=7)
        + _row(frame=1, object_id=7)
        + _row(frame=2, object_id=7),
        encoding="utf-8",
    )

    audit = audit_prediction_coverage(prediction_dir, template_zip=template_zip)
    row = audit.rows[0]

    assert not audit.ready
    assert audit.duplicate_frame_object_rows == 1
    assert audit.duplicate_frame_object_files == ["A_00.txt"]
    assert audit.blocking_reasons == ["duplicate_frame_object_rows"]
    assert row.status == "invalid"
    assert row.duplicate_frame_object_rows == 1


def test_prediction_coverage_audit_detects_invalid_class_and_visibility_rows(
    tmp_path: Path,
) -> None:
    template_zip = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_zip(template_zip, {"A_00.txt": ""})
    prediction_dir.joinpath("A_00.txt").write_text(
        _row(frame=1, object_id=1, class_id=0, visibility=1.0)
        + _row(frame=2, object_id=2, class_id=1, visibility=1.5),
        encoding="utf-8",
    )

    audit = audit_prediction_coverage(prediction_dir, template_zip=template_zip)
    row = audit.rows[0]

    assert not audit.ready
    assert audit.invalid_class_rows == 1
    assert audit.invalid_visibility_rows == 1
    assert audit.invalid_class_files == ["A_00.txt"]
    assert audit.invalid_visibility_files == ["A_00.txt"]
    assert audit.blocking_reasons == ["invalid_class_rows", "invalid_visibility_rows"]
    assert row.status == "invalid"
    assert row.invalid_class_rows == 1
    assert row.invalid_visibility_rows == 1


def test_prediction_coverage_audit_cli_writes_json_and_rows(tmp_path: Path) -> None:
    template_zip = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    output_json = tmp_path / "coverage.json"
    row_csv = tmp_path / "coverage_rows.csv"
    prediction_dir.mkdir()
    _write_zip(template_zip, {"A_00.txt": ""})
    prediction_dir.joinpath("A_00.txt").write_text(_row(), encoding="utf-8")

    status = coverage_audit_main(
        [
            str(prediction_dir),
            "--template-zip",
            str(template_zip),
            "--output-json",
            str(output_json),
            "--row-csv",
            str(row_csv),
            "--require-ready",
        ]
    )

    assert status == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["blocking_reasons"] == []
    assert payload["missing_file_count"] == 0
    assert payload["out_of_range_frame_rows"] == 0
    assert payload["duplicate_frame_object_rows"] == 0
    assert payload["invalid_class_rows"] == 0
    assert payload["invalid_visibility_rows"] == 0
    row_text = row_csv.read_text(encoding="utf-8")
    assert "expected_frame_count" in row_text
    assert "duplicate_frame_object_rows" in row_text
    assert "invalid_class_rows" in row_text
    assert "invalid_visibility_rows" in row_text
    assert "A_00.txt" in row_text


def test_prediction_coverage_audit_require_ready_exits_nonzero(tmp_path: Path) -> None:
    template_zip = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_zip(template_zip, {"A_00.txt": ""})

    with pytest.raises(SystemExit) as exc_info:
        coverage_audit_main(
            [
                str(prediction_dir),
                "--template-zip",
                str(template_zip),
                "--require-ready",
            ]
        )

    assert exc_info.value.code == 1
