from __future__ import annotations

import importlib
import zipfile
from pathlib import Path

import raft_uav.multi_uav_lts as multi_uav_lts
from raft_uav.multi_uav_lts.cli import validate_submission_zip


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def test_duplicate_key_guard_survives_package_reload(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    submission = tmp_path / "submission.zip"
    _write_zip(template, {"A_00.txt": ""})
    _write_zip(
        submission,
        {
            "A_00.txt": (
                "1,1,10,20,5,6,0.9,1,1\n"
                "1,1,11,21,5,6,0.8,1,1\n"
            ),
        },
    )

    before_reload = validate_submission_zip(submission, template_zip=template)
    assert before_reload.parse_errors == 1

    importlib.reload(multi_uav_lts)

    after_reload = validate_submission_zip(submission, template_zip=template)
    assert not after_reload.valid
    assert after_reload.parse_errors == 1
    assert after_reload.files[0].parse_errors == 1
