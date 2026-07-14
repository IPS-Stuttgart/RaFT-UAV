from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from raft_uav.multi_uav_lts.cli import validate_submission_zip


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


@pytest.mark.parametrize(
    ("row", "invalid_field"),
    [
        ("1,1,10,20,5,6,0.9,0,1\n", "invalid_class_rows"),
        ("1,1,10,20,5,6,0.9,1,-0.1\n", "invalid_visibility_rows"),
        ("1,1,10,20,5,6,0.9,1,1.1\n", "invalid_visibility_rows"),
    ],
)
def test_validate_submission_zip_reports_invalid_class_and_visibility_domains(
    tmp_path: Path,
    row: str,
    invalid_field: str,
) -> None:
    template = tmp_path / "template.zip"
    submission = tmp_path / "submission.zip"
    _write_zip(template, {"A_00.txt": ""})
    _write_zip(submission, {"A_00.txt": row})

    validation = validate_submission_zip(submission, template_zip=template)

    assert validation.valid is False
    assert getattr(validation, invalid_field) == 1
