"""Treat non-UTF-8 topic maps in archives as malformed metadata."""

from __future__ import annotations

from pathlib import Path
import tarfile
import zipfile

from raft_uav.mmuad import layout

_PATCH_MARKER = "_raft_uav_archive_topic_map_decode_guard"


def _decode_topic_map(payload: bytes) -> str | None:
    """Decode UTF-8 metadata, returning ``None`` for malformed bytes."""

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _inspect_zip_archive(
    archive_path: Path,
    root: Path,
) -> list[layout.LayoutFile]:
    """Inventory ZIP members without failing on malformed topic-map bytes."""

    rows: list[layout.LayoutFile] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = layout._normalize_archive_member_name(info.filename)
            topic_map_text = None
            if layout._is_topic_map_member(name):
                with archive.open(info) as handle:
                    topic_map_text = _decode_topic_map(handle.read())
            rows.append(
                layout._classify_archive_member(
                    archive_path,
                    root,
                    member_name=name,
                    size_bytes=int(info.file_size),
                    topic_map_text=topic_map_text,
                )
            )
    return rows


def _inspect_tar_archive(
    archive_path: Path,
    root: Path,
) -> list[layout.LayoutFile]:
    """Inventory TAR members without failing on malformed topic-map bytes."""

    rows: list[layout.LayoutFile] = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        for info in archive.getmembers():
            if not info.isfile():
                continue
            name = layout._normalize_archive_member_name(info.name)
            topic_map_text = None
            if layout._is_topic_map_member(name):
                handle = archive.extractfile(info)
                if handle is not None:
                    with handle:
                        topic_map_text = _decode_topic_map(handle.read())
            rows.append(
                layout._classify_archive_member(
                    archive_path,
                    root,
                    member_name=name,
                    size_bytes=int(info.size),
                    topic_map_text=topic_map_text,
                )
            )
    return rows


def install() -> None:
    """Install archive readers that tolerate malformed topic-map encodings."""

    if getattr(layout, _PATCH_MARKER, False):
        return
    layout._inspect_zip_archive = _inspect_zip_archive
    layout._inspect_tar_archive = _inspect_tar_archive
    setattr(layout, _PATCH_MARKER, True)
