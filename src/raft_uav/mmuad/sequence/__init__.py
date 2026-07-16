"""Compatibility fix for safe NumPy timestamp-sidecar loading.

The maintained implementation lives in the sibling ``sequence.py`` module. This
package preserves the public import path while preventing timestamp sidecars from
deserializing pickle-backed object arrays.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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


_IMPL._timestamp_map_from_numpy_sidecar = _timestamp_map_from_numpy_sidecar
_IMPL._timestamps_from_numpy_sidecar = _timestamps_from_numpy_sidecar

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_timestamp_map_from_numpy_sidecar"] = _timestamp_map_from_numpy_sidecar
globals()["_timestamps_from_numpy_sidecar"] = _timestamps_from_numpy_sidecar

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
