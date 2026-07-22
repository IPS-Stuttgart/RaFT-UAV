"""Compatibility fixes for safe MMUAD sequence discovery and loading.

The maintained implementation lives in the sibling ``sequence.py`` module. This
package preserves the public import path while preventing timestamp sidecars from
deserializing pickle-backed object arrays and preventing directory symlink cycles
from recursing indefinitely during sequence discovery.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Hashable

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "sequence.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._sequence_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load MMUAD sequence implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _directory_identity(path: Path) -> Hashable | None:
    """Return a stable identity for one directory, following symlinks safely."""

    try:
        stat_result = path.stat()
    except (OSError, RuntimeError):
        return None
    inode_identity = (stat_result.st_dev, stat_result.st_ino)
    if inode_identity != (0, 0):
        return inode_identity
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _candidate_sequence_dirs(root: Path, *, sequence_glob: str) -> list[Path]:
    """Discover sequence directories without revisiting aliased ancestors."""

    if not root.is_dir():
        return []
    candidates: list[Path] = []
    ancestor_identities: set[Hashable] = set()
    root_identity = _directory_identity(root)
    if root_identity is not None:
        ancestor_identities.add(root_identity)
    for child in _IMPL._non_modality_child_dirs(root):
        _collect_sequence_dirs(
            child,
            root=root,
            sequence_glob=sequence_glob,
            candidates=candidates,
            ancestor_identities=ancestor_identities,
        )
    return _IMPL._unique_paths(candidates)


def _collect_sequence_dirs(
    path: Path,
    *,
    root: Path,
    sequence_glob: str,
    candidates: list[Path],
    ancestor_identities: set[Hashable] | None = None,
) -> None:
    """Recursively collect sequences while breaking directory-alias cycles."""

    if _IMPL._is_modality_dir(path):
        return
    if ancestor_identities is None:
        ancestor_identities = set()
    identity = _directory_identity(path)
    if identity is None or identity in ancestor_identities:
        return
    ancestor_identities.add(identity)
    try:
        prior_count = len(candidates)
        for child in _IMPL._non_modality_child_dirs(path):
            _collect_sequence_dirs(
                child,
                root=root,
                sequence_glob=sequence_glob,
                candidates=candidates,
                ancestor_identities=ancestor_identities,
            )
        if len(candidates) > prior_count:
            return
        if _IMPL._is_sequence_wrapper_dir(path):
            return
        if _IMPL._sequence_dir_matches(
            path,
            root=root,
            sequence_glob=sequence_glob,
        ) and _IMPL._looks_like_sequence(path):
            candidates.append(path)
    finally:
        ancestor_identities.remove(identity)


def _timestamp_map_from_numpy_sidecar(path: Path) -> dict[str, float]:
    """Read a filename-to-time NumPy sidecar without loading pickled objects."""

    payload = np.load(path, allow_pickle=False)
    try:
        if isinstance(payload, np.lib.npyio.NpzFile):
            name_key = _IMPL._matching_npz_key_or_none(
                payload,
                preferred=_IMPL.TIMESTAMP_FRAME_NAME_ALIASES,
            )
            time_key = _IMPL._matching_npz_key_or_none(
                payload,
                preferred=(
                    "time_s",
                    "timestamp_s",
                    "timestamp",
                    "timestamps",
                    "times",
                    "frame_times",
                    "frame_timestamps",
                ),
            )
            if name_key is None or time_key is None:
                return {}
            return _IMPL._timestamp_map_from_name_time_arrays(
                payload[name_key],
                payload[time_key],
            )
        array = np.asarray(payload)
        if array.dtype.names:
            frame = _IMPL.pd.DataFrame({name: array[name] for name in array.dtype.names})
            return _IMPL._timestamp_map_from_frame(frame)
        return {}
    finally:
        if isinstance(payload, np.lib.npyio.NpzFile):
            payload.close()


def _timestamps_from_numpy_sidecar(path: Path) -> list[float]:
    """Read ordered NumPy timestamps without loading pickled objects."""

    payload = np.load(path, allow_pickle=False)
    try:
        if isinstance(payload, np.lib.npyio.NpzFile):
            key = _IMPL._first_npz_key(
                payload,
                preferred=(
                    "time_s",
                    "timestamp_s",
                    "timestamp",
                    "timestamps",
                    "times",
                    "frame_times",
                    "frame_timestamps",
                ),
            )
            return _IMPL._timestamps_from_numpy_array(payload[key])
        return _IMPL._timestamps_from_numpy_array(payload)
    finally:
        if isinstance(payload, np.lib.npyio.NpzFile):
            payload.close()


_IMPL._candidate_sequence_dirs = _candidate_sequence_dirs
_IMPL._collect_sequence_dirs = _collect_sequence_dirs
_IMPL._timestamp_map_from_numpy_sidecar = _timestamp_map_from_numpy_sidecar
_IMPL._timestamps_from_numpy_sidecar = _timestamps_from_numpy_sidecar

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_candidate_sequence_dirs"] = _candidate_sequence_dirs
globals()["_collect_sequence_dirs"] = _collect_sequence_dirs
globals()["_directory_identity"] = _directory_identity
globals()["_timestamp_map_from_numpy_sidecar"] = _timestamp_map_from_numpy_sidecar
globals()["_timestamps_from_numpy_sidecar"] = _timestamps_from_numpy_sidecar

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
