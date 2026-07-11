from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from raft_uav.multi_uav_lts.cli import validate_submission_zip


def test_submission_validator_rejects_duplicate_members_that_pad_file_count(
    tmp_path: Path,
) -> None:
    submission_zip = tmp_path / "submission.zip"
    with zipfile.ZipFile(submission_zip, "w") as archive:
        for index in range(97):
            archive.writestr(f"sequence_{index:03d}.txt", "")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("sequence_000.txt", "")

    validation = validate_submission_zip(submission_zip, expected_file_count=98)

    assert validation.file_count == 98
    assert validation.duplicate_entries == ["sequence_000.txt"]
    assert len(validation.files) == 97
    assert not validation.valid


def test_submission_validator_accepts_unique_members(tmp_path: Path) -> None:
    submission_zip = tmp_path / "submission.zip"
    with zipfile.ZipFile(submission_zip, "w") as archive:
        archive.writestr("sequence_a.txt", "")
        archive.writestr("sequence_b.txt", "")

    validation = validate_submission_zip(submission_zip, expected_file_count=2)

    assert validation.valid
    assert validation.duplicate_entries == []
    assert len(validation.files) == 2
