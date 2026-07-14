from __future__ import annotations

import io
from pathlib import Path
import tarfile
import zipfile

from raft_uav.mmuad.archive import extract_mmuad_archive


def _assert_normalized_path_collision_is_skipped(
    archive_path: Path,
    output_root: Path,
) -> None:
    manifest = extract_mmuad_archive(archive_path, output_root)

    assert manifest["member_count"] == 2
    assert manifest["extracted_file_count"] == 1
    assert manifest["skipped_member_count"] == 1
    assert manifest["skipped_members"] == [
        {"member": "sequence/frame.txt", "reason": "duplicate_member_path"}
    ]
    extracted_path = Path(manifest["extracted_files"][0]["path"])
    assert extracted_path.read_text(encoding="utf-8") == "first"


def test_zip_normalized_member_collision_does_not_overwrite(tmp_path: Path) -> None:
    archive_path = tmp_path / "sequences.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("sequence/frame.txt", "first")
        archive.writestr(r"sequence\frame.txt", "second")

    _assert_normalized_path_collision_is_skipped(archive_path, tmp_path / "zip-output")


def test_tar_normalized_member_collision_does_not_overwrite(tmp_path: Path) -> None:
    archive_path = tmp_path / "sequences.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        for member_name, payload in (
            ("sequence/frame.txt", b"first"),
            (r"sequence\frame.txt", b"second"),
        ):
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    _assert_normalized_path_collision_is_skipped(archive_path, tmp_path / "tar-output")
