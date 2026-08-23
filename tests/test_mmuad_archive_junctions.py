from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

import raft_uav.mmuad.archive as archive_module
from raft_uav.mmuad.archive import extract_mmuad_archive


def _write_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sequence/data.txt", "payload")


def _deterministic_root(archive: Path, output_root: Path) -> Path:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return output_root / f"bundle-{digest[:12]}"


def test_extract_mmuad_archive_rejects_junctioned_extraction_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "bundle.zip"
    _write_archive(archive)
    output_root = tmp_path / "output"
    output_root.mkdir()
    extraction_root = _deterministic_root(archive, output_root)

    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == extraction_root,
        raising=False,
    )

    with pytest.raises(ValueError, match="extraction root junction"):
        extract_mmuad_archive(archive, output_root)

    assert not (extraction_root / "sequence" / "data.txt").exists()


def test_extract_mmuad_archive_rejects_junctioned_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "bundle.zip"
    _write_archive(archive)
    output_root = tmp_path / "output"
    output_root.mkdir()

    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == output_root,
        raising=False,
    )

    with pytest.raises(ValueError, match="output root junction"):
        extract_mmuad_archive(archive, output_root)

    extraction_root = _deterministic_root(archive, output_root)
    assert not (extraction_root / "sequence" / "data.txt").exists()


def test_directory_link_kind_detects_pre_python_312_junction_metadata() -> None:
    class LegacyJunctionPath:
        def is_symlink(self) -> bool:
            return False

        def lstat(self) -> SimpleNamespace:
            return SimpleNamespace(
                st_reparse_tag=archive_module._IO_REPARSE_TAG_MOUNT_POINT
            )

    assert archive_module._directory_link_kind(LegacyJunctionPath()) == "junction"
