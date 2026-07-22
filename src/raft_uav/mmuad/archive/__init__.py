"""Compatibility wrapper protecting deterministic archive extraction roots.

The maintained implementation lives in the sibling ``archive.py`` module. This
package preserves the public import path while rejecting a pre-existing symlink
at the hash-derived extraction directory before any archive member is written.
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


class _ArchiveModule(ModuleType):
    """Module proxy that keeps runtime monkeypatches visible to legacy globals."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "_IMPL":
            return
        implementation = self.__dict__.get("_IMPL")
        if implementation is not None and hasattr(implementation, name):
            setattr(implementation, name, value)


def extract_mmuad_archive(archive_path: Path, output_root: Path) -> dict[str, Any]:
    """Extract an archive without following a pre-existing extraction-root link."""

    archive = Path(archive_path)
    if not archive.is_file() or _IMPL.archive_kind(archive) == "unknown":
        return _ORIGINAL_EXTRACT_MMUAD_ARCHIVE(archive, output_root)

    output = Path(output_root)
    archive_sha256 = _IMPL._sha256_file(archive)
    extract_root = output / (
        f"{_IMPL._safe_dir_name(_IMPL.archive_stem(archive))}-{archive_sha256[:12]}"
    )
    if extract_root.is_symlink():
        raise ValueError(f"unsafe MMUAD extraction root symlink: {extract_root}")

    output.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(exist_ok=True)
    expected_root = output.resolve() / extract_root.name
    if extract_root.is_symlink() or extract_root.resolve() != expected_root:
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
