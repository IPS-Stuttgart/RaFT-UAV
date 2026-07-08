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


def _read_text_csv(path: Path) -> pd.DataFrame:
    """Read CSV exports without coercing opaque ids or keeping padded headers."""

    if path.suffix.lower() == ".tsv":
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_candidate_csv(
    path: Path,
    *,
    default_sequence_id: str = "default",
    source: str = "candidate",
) -> _impl.CandidateFrame:
    """Load a normalized or alias-compatible candidate CSV without coercing ids."""

    raw = _read_text_csv(Path(path))
    if _impl._should_add_default_candidate_source(raw, source=source):
        raw["source"] = source
    rows = _impl.normalize_candidate_columns(
        raw,
        default_sequence_id=default_sequence_id,
        default_source=source,
    )
    frame = _impl.CandidateFrame(rows)
    frame.validate()
    return frame


def load_truth_csv(path: Path, *, default_sequence_id: str = "default") -> _impl.TruthFrame:
    """Load a normalized or alias-compatible truth CSV without coercing ids."""

    rows = _impl.normalize_truth_columns(
        _read_text_csv(Path(path)),
        default_sequence_id=default_sequence_id,
    )
    frame = _impl.TruthFrame(rows)
    frame.validate()
    return frame


def load_point_cloud_csv_as_points(
    path: Path,
    *,
    source: str = "lidar-cluster",
) -> pd.DataFrame:
    """Load a point-cloud CSV row table without clustering it."""

    points = _impl.normalize_truth_columns(_read_text_csv(Path(path)))
    if "source" not in points.columns:
        points["source"] = source
    else:
        points["source"] = _impl._filled_text_series(points["source"], default=source)
    return points


def _read_point_cloud_csv(path: Path) -> pd.DataFrame:
    frame = _read_text_csv(Path(path))
    try:
        return _impl.normalize_truth_columns(frame)
    except ValueError as exc:
        if "time_s" not in str(exc):
            raise
    return _impl._normalize_point_frame(frame, path=path)


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


_impl.load_candidate_csv = load_candidate_csv
_impl.load_truth_csv = load_truth_csv
_impl.load_point_cloud_csv_as_points = load_point_cloud_csv_as_points
_impl._read_point_cloud_csv = _read_point_cloud_csv
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
