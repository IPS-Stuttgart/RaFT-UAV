from __future__ import annotations

from pathlib import Path

import numpy as np

from raft_uav.mmuad.io import (
    load_point_cloud_file_as_candidates,
    load_point_cloud_file_as_points,
)


def test_numpy_xyzi_uses_intensity_without_splitting_the_frame(tmp_path: Path) -> None:
    path = tmp_path / "livox_12.5.npy"
    points = np.array(
        [
            [0.0, 0.0, 0.0, 0.1],
            [0.1, 0.1, 0.0, 0.2],
            [0.2, 0.2, 0.0, 0.3],
        ],
        dtype=float,
    )
    np.save(path, points)

    frame = load_point_cloud_file_as_points(path, source="livox")

    assert frame["time_s"].tolist() == [12.5, 12.5, 12.5]
    np.testing.assert_allclose(frame["intensity"].to_numpy(dtype=float), points[:, 3])

    candidates = load_point_cloud_file_as_candidates(
        path,
        source="livox",
        voxel_size_m=1.0,
        min_points=3,
    ).rows

    assert len(candidates) == 1
    assert candidates["time_s"].tolist() == [12.5]
