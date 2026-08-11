"""Compatibility wrapper protecting deterministic archive extraction roots.

The maintained implementation lives in the sibling ``archive.py`` module. This
package preserves the public import path while rejecting pre-existing directory
links at the output and hash-derived extraction directories before any archive
member is written.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

_IMPL_PATH = Path(__file__).resolve().parent.parent / "archive.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._archive_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load MMUAD archive implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_EXTRACT_MMUAD_ARCHIVE = _IMPL.extract_mmuad_archive
_IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003


class _ArchiveModule(ModuleType):
    """Module proxy that keeps runtime monkeypatches visible to legacy globals."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "_IMPL":
            return
        implementation = self.__dict__.get("_IMPL")
        if implementation is not None and hasattr(implementation, name):
            setattr(implementation, name, value)


def _directory_link_kind(path: Path) -> str | None:
    """Return the directory-link kind for a symlink or Windows junction."""

    if path.is_symlink():
        return "symlink"

    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return "junction"

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if getattr(metadata, "st_reparse_tag", None) == _IO_REPARSE_TAG_MOUNT_POINT:
        return "junction"
    return None


def _reject_directory_link(path: Path, *, label: str) -> None:
    kind = _directory_link_kind(path)
    if kind is not None:
        raise ValueError(f"unsafe MMUAD {label} {kind}: {path}")


def extract_mmuad_archive(archive_path: Path, output_root: Path) -> dict[str, Any]:
    """Extract an archive without following pre-existing output-directory links."""

    archive = Path(archive_path)
    if not archive.is_file() or _IMPL.archive_kind(archive) == "unknown":
        return _ORIGINAL_EXTRACT_MMUAD_ARCHIVE(archive, output_root)

    output = Path(output_root)
    _reject_directory_link(output, label="output root")

    archive_sha256 = _IMPL._sha256_file(archive)
    extract_root = output / (
        f"{_IMPL._safe_dir_name(_IMPL.archive_stem(archive))}-{archive_sha256[:12]}"
    )
    _reject_directory_link(extract_root, label="extraction root")

    output.mkdir(parents=True, exist_ok=True)
    _reject_directory_link(output, label="output root")
    extract_root.mkdir(exist_ok=True)
    expected_root = output.resolve() / extract_root.name
    _reject_directory_link(extract_root, label="extraction root")
    if extract_root.resolve() != expected_root:
        raise ValueError(f"unsafe MMUAD extraction root: {extract_root}")

    return _ORIGINAL_EXTRACT_MMUAD_ARCHIVE(archive, output)


_IMPL.extract_mmuad_archive = extract_mmuad_archive

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["extract_mmuad_archive"] = extract_mmuad_archive
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
sys.modules[__name__].__class__ = _ArchiveModule
