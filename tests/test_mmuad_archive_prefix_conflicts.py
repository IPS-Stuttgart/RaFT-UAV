from __future__ import annotations

import io
from pathlib import Path
import tarfile
import zipfile

import pytest

from raft_uav.mmuad.archive import extract_mmuad_archive


def _write_archive(archive_path: Path, members: list[tuple[str, bytes]]) -> None:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, mode="w") as archive:
            for member_name, payload in members:
                archive.writestr(member_name, payload)
        return

    with tarfile.open(archive_path, mode="w") as archive:
        for member_name, payload in members:
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


@pytest.mark.parametrize("suffix", [".zip", ".tar"])
@pytest.mark.parametrize(
    ("members", "extracted_member", "extracted_payload", "skipped_member"),
    [
        (
            [("sequence", b"parent"), ("sequence/frame.txt", b"child")],
            "sequence",
            b"parent",
            "sequence/frame.txt",
        ),
        (
            [("sequence/frame.txt", b"child"), ("sequence", b"parent")],
            "sequence/frame.txt",
            b"child",
            "sequence",
        ),
    ],
    ids=["file-before-child", "child-before-file"],
)
def test_archive_prefix_path_conflicts_are_skipped(
    tmp_path: Path,
    suffix: str,
    members: list[tuple[str, bytes]],
    extracted_member: str,
    extracted_payload: bytes,
    skipped_member: str,
) -> None:
    archive_path = tmp_path / f"sequences{suffix}"
    _write_archive(archive_path, members)

    manifest = extract_mmuad_archive(archive_path, tmp_path / "output")

    assert manifest["member_count"] == 2
    assert manifest["extracted_file_count"] == 1
    assert manifest["skipped_member_count"] == 1
    assert manifest["extracted_files"][0]["member"] == extracted_member
    assert Path(manifest["extracted_files"][0]["path"]).read_bytes() == extracted_payload
    assert manifest["skipped_members"] == [
        {"member": skipped_member, "reason": "conflicting_member_path"}
    ]
