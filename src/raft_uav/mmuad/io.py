"""Compatibility wrapper around the MMUAD I/O implementation.

The implementation is kept in :mod:`raft_uav.mmuad._io_impl`.  This module
installs the gzip-aware NumPy readers before re-exporting the public and
internal helpers from the implementation module.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad import _io_impl as _impl

_MODULE_METADATA_NAMES = {
    "__builtins__",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
}


def _numpy_array_from_export(path: Path, *, preferred_keys: Iterable[str]) -> np.ndarray:
    """Load a plain or gzip-compressed ``.npy``/``.npz`` export eagerly."""

    path = Path(path)
    payload = np.load(BytesIO(_impl.read_binary_export(path)), allow_pickle=False)
    try:
        if isinstance(payload, np.lib.npyio.NpzFile):
            if not payload.files:
                raise ValueError(f"NumPy archive {path} does not contain arrays")
            key = next(
                (candidate for candidate in preferred_keys if candidate in payload.files),
                payload.files[0],
            )
            return np.asarray(payload[key])
        return np.asarray(payload)
    finally:
        if isinstance(payload, np.lib.npyio.NpzFile):
            payload.close()


def _read_numpy_trajectory_table(path: Path) -> pd.DataFrame:
    """Read compact NumPy trajectory rows used by candidate/truth loaders."""

    arr = _numpy_array_from_export(
        path,
        preferred_keys=(
            "trajectory",
            "trajectories",
            "truth",
            "ground_truth",
            "gt",
            "candidates",
            "detections",
            "tracks",
            "poses",
            "rows",
            "data",
        ),
    )
    if arr.size == 0:
        return pd.DataFrame(
            {
                "time_s": pd.Series(dtype=float),
                "x_m": pd.Series(dtype=float),
                "y_m": pd.Series(dtype=float),
                "z_m": pd.Series(dtype=float),
            }
        )
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        compact = arr.reshape(-1)
        if compact.size >= 3:
            frame = pd.DataFrame([compact[:3]], columns=["x_m", "y_m", "z_m"])
            frame.insert(0, "time_s", _impl.infer_time_s_from_filename(path))
            if compact.shape[0] >= 4:
                frame["confidence"] = compact[3]
            return frame
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"NumPy trajectory table must be shape (N, >=3), got {arr.shape}")
    if arr.shape[1] == 3:
        frame = pd.DataFrame({"x_m": arr[:, 0], "y_m": arr[:, 1], "z_m": arr[:, 2]})
        frame.insert(0, "time_s", _impl.infer_time_s_from_filename(path))
        return frame
    frame = pd.DataFrame(
        {
            "time_s": arr[:, 0],
            "x_m": arr[:, 1],
            "y_m": arr[:, 2],
            "z_m": arr[:, 3],
        }
    )
    if arr.shape[1] >= 5:
        frame["confidence"] = arr[:, 4]
    return frame


def _read_numpy_point_cloud(path: Path) -> pd.DataFrame:
    """Read compact NumPy point-cloud rows from plain or gzipped exports."""

    arr = _numpy_array_from_export(
        path,
        preferred_keys=(
            "points",
            "point_cloud",
            "pointcloud",
            "cloud",
            "lidar_points",
            "livox_points",
            "rows",
            "data",
        ),
    )
    if arr.size == 0:
        return pd.DataFrame(
            {
                "x_m": pd.Series(dtype=float),
                "y_m": pd.Series(dtype=float),
                "z_m": pd.Series(dtype=float),
            }
        )
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"NumPy point cloud must be shape (N, >=3), got {arr.shape}")
    frame = pd.DataFrame({"x_m": arr[:, 0], "y_m": arr[:, 1], "z_m": arr[:, 2]})
    if arr.shape[1] >= 4:
        frame["time_s"] = arr[:, 3]
    return _impl._normalize_point_frame(frame, path=path)


_impl._numpy_array_from_export = _numpy_array_from_export
_impl._read_numpy_trajectory_table = _read_numpy_trajectory_table
_impl._read_numpy_point_cloud = _read_numpy_point_cloud

globals().update(
    {
        name: value
        for name, value in vars(_impl).items()
        if name not in _MODULE_METADATA_NAMES
    }
)
__doc__ = _impl.__doc__
__all__ = [name for name in globals() if not name.startswith("__")]
