from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import pytest

from raft_uav.mmuad.archive import extract_mmuad_archive


def _write_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sequence/data.txt", "payload")


def _deterministic_root(archive: Path, output_root: Path) -> Path:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return output_root / f"bundle-{digest[:12]}"


def test_extract_mmuad_archive_rejects_symlinked_root(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    _write_archive(archive)
    output_root = tmp_path / "output"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    extraction_root = _deterministic_root(archive, output_root)
    try:
        extraction_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform permission limitation
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="extraction root symlink"):
        extract_mmuad_archive(archive, output_root)

    assert not (outside / "sequence" / "data.txt").exists()
