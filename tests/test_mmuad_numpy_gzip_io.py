from __future__ import annotations

import gzip
from io import BytesIO
from pathlib import Path

import numpy as np

from raft_uav.mmuad.io import load_point_cloud_file_as_points, load_truth_file


def _write_gzipped_npy(path: Path, array: np.ndarray) -> None:
    buffer = BytesIO()
    np.save(buffer, array)
    path.write_bytes(gzip.compress(buffer.getvalue()))


def _write_gzipped_npz(path: Path, **arrays: np.ndarray) -> None:
    buffer = BytesIO()
    np.savez(buffer, **arrays)
    path.write_bytes(gzip.compress(buffer.getvalue()))


def test_load_truth_file_accepts_gzipped_npy_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "truth.npy.gz"
    rows = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 4.0, 5.0, 6.0],
        ],
        dtype=float,
    )
    _write_gzipped_npy(path, rows)

    truth = load_truth_file(path, default_sequence_id="seq-gz").rows

    assert truth["sequence_id"].tolist() == ["seq-gz", "seq-gz"]
    np.testing.assert_allclose(
        truth[["time_s", "x_m", "y_m", "z_m"]].to_numpy(dtype=float),
        rows,
    )


def test_load_point_cloud_file_as_points_accepts_gzipped_npz(tmp_path: Path) -> None:
    path = tmp_path / "livox_points.npz.gz"
    points = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )
    _write_gzipped_npz(path, points=points)

    frame = load_point_cloud_file_as_points(
        path,
        source="livox",
        sequence_id="seq-gz",
        time_s=12.5,
    )

    assert frame["sequence_id"].tolist() == ["seq-gz", "seq-gz"]
    assert frame["source"].tolist() == ["livox", "livox"]
    assert frame["time_s"].tolist() == [12.5, 12.5]
    np.testing.assert_allclose(frame[["x_m", "y_m", "z_m"]].to_numpy(dtype=float), points)
