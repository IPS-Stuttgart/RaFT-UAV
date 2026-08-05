"""Compatibility fix for malformed archived MMUAD topic-map metadata.

The maintained implementation lives in the sibling ``inspect.py`` module. This
package preserves the public import path while treating non-UTF-8 topic-map
members as malformed metadata instead of aborting the complete archive inventory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_IMPL_PATH = Path(__file__).resolve().parent.parent / "inspect.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._inspect_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load inspect implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _decode_topic_map_text(payload: bytes) -> str | None:
    """Decode archive metadata, returning ``None`` for malformed UTF-8."""

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _inspect_zip_archive(archive_path: Path, root: Path) -> list[object]:
    rows: list[object] = []
    with _IMPL.zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = _IMPL._normalize_archive_member_name(info.filename)
            topic_map_text = None
            if _IMPL._is_topic_map_member(name):
                with archive.open(info) as handle:
                    topic_map_text = _decode_topic_map_text(handle.read())
            rows.append(
                _IMPL._archive_member_record(
                    archive_path,
                    root,
                    member_name=name,
                    size_bytes=int(info.file_size),
                    topic_map_text=topic_map_text,
                )
            )
    return rows


def _inspect_tar_archive(archive_path: Path, root: Path) -> list[object]:
    rows: list[object] = []
    with _IMPL.tarfile.open(archive_path, mode="r:*") as archive:
        for info in archive.getmembers():
            if not info.isfile():
                continue
            name = _IMPL._normalize_archive_member_name(info.name)
            topic_map_text = None
            if _IMPL._is_topic_map_member(name):
                handle = archive.extractfile(info)
                if handle is not None:
                    with handle:
                        topic_map_text = _decode_topic_map_text(handle.read())
            rows.append(
                _IMPL._archive_member_record(
                    archive_path,
                    root,
                    member_name=name,
                    size_bytes=int(info.size),
                    topic_map_text=topic_map_text,
                )
            )
    return rows


_IMPL._decode_topic_map_text = _decode_topic_map_text
_IMPL._inspect_zip_archive = _inspect_zip_archive
_IMPL._inspect_tar_archive = _inspect_tar_archive

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_decode_topic_map_text"] = _decode_topic_map_text
globals()["_inspect_zip_archive"] = _inspect_zip_archive
globals()["_inspect_tar_archive"] = _inspect_tar_archive

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
