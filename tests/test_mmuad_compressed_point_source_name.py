from __future__ import annotations

import gzip
from io import BytesIO
from pathlib import Path

import numpy as np

from raft_uav.mmuad.io import (
    load_point_cloud_file_as_candidates,
    load_point_cloud_file_as_points,
)


def _write_gzipped_npz(path: Path, **arrays: np.ndarray) -> None:
    buffer = BytesIO()
    np.savez(buffer, **arrays)
    path.write_bytes(gzip.compress(buffer.getvalue()))


def test_gzipped_point_cloud_infers_source_without_inner_extension(
    tmp_path: Path,
) -> None:
    path = tmp_path / "livox_points.npz.gz"
    _write_gzipped_npz(path, points=np.array([[1.0, 2.0, 3.0]], dtype=float))

    frame = load_point_cloud_file_as_points(
        path,
        sequence_id="seq-gz",
        time_s=12.5,
    )

    assert frame["source"].tolist() == ["livox-cluster"]


def test_gzipped_point_candidates_preserve_normalized_inferred_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "livox_points.npz.gz"
    _write_gzipped_npz(path, points=np.array([[1.0, 2.0, 3.0]], dtype=float))

    frame = load_point_cloud_file_as_candidates(
        path,
        sequence_id="seq-gz",
        time_s=12.5,
        min_points=1,
    ).rows

    assert frame["source"].tolist() == ["livox-cluster"]
