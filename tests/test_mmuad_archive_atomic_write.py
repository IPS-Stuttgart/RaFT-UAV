from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

import raft_uav.mmuad.archive as archive_module


@pytest.mark.parametrize("existing_contents", [None, b"previous complete contents"])
def test_failed_archive_copy_does_not_publish_partial_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_contents: bytes | None,
) -> None:
    archive_path = tmp_path / "input.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("sequence/data.csv", b"complete archive contents")

    extract_root = tmp_path / "extracted"
    destination = extract_root / "sequence" / "data.csv"
    if existing_contents is not None:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(existing_contents)

    def fail_after_partial_write(source, target) -> None:
        del source
        target.write(b"partial")
        raise OSError("simulated archive read failure")

    monkeypatch.setattr(archive_module.shutil, "copyfileobj", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated archive read failure"):
        archive_module._extract_zip_archive(archive_path, extract_root)

    if existing_contents is None:
        assert not destination.exists()
    else:
        assert destination.read_bytes() == existing_contents
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))
