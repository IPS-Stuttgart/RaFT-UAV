"""Safe archive helpers for MMUAD sequence-root onboarding.

These helpers support downloaded/exported ZIP and TAR-family bundles without
claiming to parse undocumented native packet formats.  Archives are extracted
into a controlled output directory before the normal sequence-root loader runs.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import re
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import Any
import zipfile


ZIP_SUFFIXES = {".zip"}
TAR_SUFFIXES = {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}
ARCHIVE_SUFFIXES = ZIP_SUFFIXES | TAR_SUFFIXES
ARCHIVE_EXTRACTION_SCHEMA = "raft-uav-mmuad-archive-extraction-v1"


def is_supported_archive(path: Path) -> bool:
    """Return true when ``path`` is a ZIP or TAR-family archive."""

    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def archive_kind(path: Path) -> str:
    """Return ``zip``, ``tar``, or ``unknown`` for a supported archive path."""

    name = Path(path).name.lower()
    if any(name.endswith(suffix) for suffix in ZIP_SUFFIXES):
        return "zip"
    if any(name.endswith(suffix) for suffix in TAR_SUFFIXES):
        return "tar"
    return "unknown"


def archive_stem(path: Path) -> str:
    """Return a stable archive stem with multi-part suffixes removed."""

    name = Path(path).name
    lower = name.lower()
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def extract_mmuad_archive(archive_path: Path, output_root: Path) -> dict[str, Any]:
    """Extract a supported archive under ``output_root`` and return a manifest.

    The extractor rejects absolute paths, parent-directory traversal, drive-like
    member names, and archive links.  Existing files with the same archive hash
    may be overwritten, but extraction is confined to the returned root.
    """

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    kind = archive_kind(archive_path)
    if kind == "unknown":
        raise ValueError(f"unsupported MMUAD archive type: {archive_path}")
    archive_sha256 = _sha256_file(archive_path)
    output_root = Path(output_root)
    extract_root = output_root / f"{_safe_dir_name(archive_stem(archive_path))}-{archive_sha256[:12]}"
    extract_root.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        extracted, skipped = _extract_zip_archive(archive_path, extract_root)
    else:
        extracted, skipped = _extract_tar_archive(archive_path, extract_root)
    return {
        "schema": ARCHIVE_EXTRACTION_SCHEMA,
        "archive_path": str(archive_path),
        "archive_format": kind,
        "archive_size_bytes": int(archive_path.stat().st_size),
        "archive_sha256": archive_sha256,
        "extract_root": str(extract_root),
        "member_count": len(extracted) + len(skipped),
        "extracted_file_count": len(extracted),
        "skipped_member_count": len(skipped),
        "extracted_files": extracted,
        "skipped_members": skipped,
        "total_uncompressed_size_bytes": int(sum(item["size_bytes"] for item in extracted)),
    }


def _extract_zip_archive(archive_path: Path, extract_root: Path) -> tuple[list[dict], list[dict]]:
    extracted: list[dict] = []
    skipped: list[dict] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member_name = normalize_archive_member_name(info.filename)
            if _zip_member_is_link(info):
                skipped.append({"member": member_name, "reason": "link_member"})
                continue
            destination = _safe_destination(extract_root, member_name)
            if destination is None:
                skipped.append({"member": member_name, "reason": "unsafe_member_path"})
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(
                {
                    "member": member_name,
                    "path": str(destination),
                    "size_bytes": int(info.file_size),
                }
            )
    return extracted, skipped


def _extract_tar_archive(archive_path: Path, extract_root: Path) -> tuple[list[dict], list[dict]]:
    extracted: list[dict] = []
    skipped: list[dict] = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        for info in archive.getmembers():
            if not info.isfile():
                if info.isdir():
                    continue
                skipped.append({"member": normalize_archive_member_name(info.name), "reason": "link_member"})
                continue
            member_name = normalize_archive_member_name(info.name)
            destination = _safe_destination(extract_root, member_name)
            if destination is None:
                skipped.append({"member": member_name, "reason": "unsafe_member_path"})
                continue
            source = archive.extractfile(info)
            if source is None:
                skipped.append({"member": member_name, "reason": "unreadable_member"})
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(
                {
                    "member": member_name,
                    "path": str(destination),
                    "size_bytes": int(info.size),
                }
            )
    return extracted, skipped


def normalize_archive_member_name(name: str) -> str:
    """Return a POSIX-style archive member name."""

    return str(PurePosixPath(str(name).replace("\\", "/")))


def _safe_destination(root: Path, member_name: str) -> Path | None:
    member = PurePosixPath(normalize_archive_member_name(member_name))
    if member.is_absolute() or not member.parts:
        return None
    if any(part in {"", ".", ".."} for part in member.parts):
        return None
    if any(":" in part for part in member.parts):
        return None
    destination = (root / Path(*member.parts)).resolve()
    root_resolved = root.resolve()
    try:
        destination.relative_to(root_resolved)
    except ValueError:
        return None
    return destination


def _zip_member_is_link(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def _safe_dir_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "mmuad_archive"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in _iter_chunks(handle):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_chunks(handle: Any, *, size: int = 1024 * 1024) -> Iterator[bytes]:
    while True:
        chunk = handle.read(size)
        if not chunk:
            break
        yield chunk
