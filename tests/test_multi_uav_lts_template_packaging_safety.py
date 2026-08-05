from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from raft_uav.multi_uav_lts.cli import package_submission


def _write_zip(path: Path, members: list[tuple[str, str]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in members:
            archive.writestr(name, text)


def test_package_submission_rejects_template_parent_traversal(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    output_zip = tmp_path / "submission.zip"
    prediction_dir.mkdir()
    outside_prediction = tmp_path / "secret.txt"
    outside_prediction.write_text("must not be packaged", encoding="utf-8")
    _write_zip(template, [("../secret.txt", "")])

    with pytest.raises(ValueError, match="unsupported prediction members"):
        package_submission(prediction_dir, output_zip, template_zip=template)

    assert not output_zip.exists()


@pytest.mark.parametrize(
    "member_name",
    ["nested/A_00.txt", "..\\A_00.txt", "/A_00.txt", "C:\\A_00.txt"],
)
def test_package_submission_rejects_non_root_template_members(
    tmp_path: Path,
    member_name: str,
) -> None:
    template = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    output_zip = tmp_path / "submission.zip"
    prediction_dir.mkdir()
    _write_zip(template, [(member_name, "")])

    with pytest.raises(ValueError, match="unsupported prediction members"):
        package_submission(prediction_dir, output_zip, template_zip=template)

    assert not output_zip.exists()


def test_package_submission_rejects_duplicate_template_members(tmp_path: Path) -> None:
    template = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    output_zip = tmp_path / "submission.zip"
    prediction_dir.mkdir()
    _write_zip(template, [("A_00.txt", ""), ("A_00.txt", "")])

    with pytest.raises(ValueError, match="duplicate members"):
        package_submission(prediction_dir, output_zip, template_zip=template)

    assert not output_zip.exists()
