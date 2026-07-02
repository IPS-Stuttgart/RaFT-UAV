from __future__ import annotations

import zipfile
from pathlib import Path

from raft_uav.multi_uav_lts.duplicate_audit import audit_duplicate_predictions


def _row(frame: int, object_id: int) -> str:
    return f"{frame},{object_id},10,20,5,6,0.9,1,1\n"


def test_duplicate_prediction_audit_includes_nested_zip_entries(tmp_path: Path) -> None:
    submission = tmp_path / "submission.zip"
    with zipfile.ZipFile(submission, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nested/S_00.txt", _row(1, 1) + _row(1, 1))

    audit = audit_duplicate_predictions(submission)

    assert not audit.clean
    assert audit.file_count == 1
    assert audit.duplicate_files == ["nested/S_00.txt"]
    assert audit.duplicate_rows == 1
