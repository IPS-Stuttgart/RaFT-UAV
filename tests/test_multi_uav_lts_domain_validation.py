from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.cli import main as lts_main
from raft_uav.multi_uav_lts.cli import validate_submission_zip


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def test_validate_submission_rejects_invalid_class_and_visibility(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    submission = tmp_path / "submission.zip"
    _write_zip(template, {"A_00.txt": ""})
    _write_zip(
        submission,
        {
            "A_00.txt": (
                "1,1,10,20,5,6,0.9,0,1\n"
                "2,1,10,20,5,6,0.8,1,1.5\n"
            ),
        },
    )

    validation = validate_submission_zip(submission, template_zip=template)

    assert not validation.valid
    assert validation.parse_errors == 2
    assert validation.files[0].parse_errors == 2

    with pytest.raises(SystemExit) as exc_info:
        lts_main(["validate-submission", str(submission), "--template-zip", str(template)])

    assert exc_info.value.code == 1
