from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from raft_uav.multi_uav_lts.cli import normalize_prediction_text, validate_submission_zip


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def _row(frame_id: str) -> str:
    return f"{frame_id},2,10,20,5,6,1,1,1\n"


def test_normalize_prediction_text_preserves_large_integer_ids_exactly() -> None:
    large_frame_id = "9007199254740993"

    assert normalize_prediction_text(_row(large_frame_id)) == _row(large_frame_id)


def test_normalize_prediction_text_rejects_fractional_ids_above_float_precision() -> None:
    with pytest.raises(ValueError, match="integer-like"):
        normalize_prediction_text(_row("9007199254740992.5"))


def test_validate_submission_zip_rejects_fractional_ids_above_float_precision(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission.zip"
    _write_zip(submission, {"S_00.txt": _row("9007199254740992.5")})

    validation = validate_submission_zip(submission, expected_file_count=1)

    assert not validation.valid
    assert validation.parse_errors == 1
