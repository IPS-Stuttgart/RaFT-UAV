from __future__ import annotations

import zipfile
from pathlib import Path

from raft_uav.multi_uav_lts.cli import validate_submission_zip


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def test_validate_submission_zip_rejects_unsorted_rows(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    submission = tmp_path / "submission.zip"
    _write_zip(template, {"A_00.txt": ""})
    _write_zip(
        submission,
        {
            "A_00.txt": (
                "2,1,10,20,5,6,1,1,1\n"
                "1,1,10,20,5,6,1,1,1\n"
            )
        },
    )

    validation = validate_submission_zip(submission, template_zip=template)

    assert validation.unsorted_rows == 1
    assert not validation.valid
