"""Compatibility guard for malformed archived MMUAD topic-map text."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _decode_utf8_or_none(payload: bytes) -> str | None:
    """Decode UTF-8 text, returning ``None`` for malformed byte streams."""

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def patch_module(layout_module: Any) -> None:
    """Patch archive inspection so malformed topic-map text stays inventoryable."""

    def _inspect_zip_archive(archive_path: Path, root: Path) -> list[Any]:
        rows: list[Any] = []
        with layout_module.zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = layout_module._normalize_archive_member_name(info.filename)
                topic_map_text = None
                if layout_module._is_topic_map_member(name):
                    with archive.open(info) as handle:
                        topic_map_text = _decode_utf8_or_none(handle.read())
                rows.append(
                    layout_module._classify_archive_member(
                        archive_path,
                        root,
                        member_name=name,
                        size_bytes=int(info.file_size),
                        topic_map_text=topic_map_text,
                    )
                )
        return rows

    def _inspect_tar_archive(archive_path: Path, root: Path) -> list[Any]:
        rows: list[Any] = []
        with layout_module.tarfile.open(archive_path, mode="r:*") as archive:
            for info in archive.getmembers():
                if not info.isfile():
                    continue
                name = layout_module._normalize_archive_member_name(info.name)
                topic_map_text = None
                if layout_module._is_topic_map_member(name):
                    handle = archive.extractfile(info)
                    if handle is not None:
                        with handle:
                            topic_map_text = _decode_utf8_or_none(handle.read())
                rows.append(
                    layout_module._classify_archive_member(
                        archive_path,
                        root,
                        member_name=name,
                        size_bytes=int(info.size),
                        topic_map_text=topic_map_text,
                    )
                )
        return rows

    layout_module._inspect_zip_archive = _inspect_zip_archive
    layout_module._inspect_tar_archive = _inspect_tar_archive
