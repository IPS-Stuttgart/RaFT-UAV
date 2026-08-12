from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from raft_uav.mmuad.io import load_point_cloud_file_as_points


def test_binary_big_endian_ply_uses_native_pandas_buffers(tmp_path: Path) -> None:
    path = tmp_path / "frame.ply"
    header = "\n".join(
        (
            "ply",
            "format binary_big_endian 1.0",
            "element vertex 2",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
            "",
        )
    ).encode("ascii")
    payload = struct.pack(
        ">ffffff",
        1.25,
        2.5,
        3.75,
        -4.0,
        5.5,
        6.25,
    )
    path.write_bytes(header + payload)

    points = load_point_cloud_file_as_points(path, source="test-lidar")

    coordinate_columns = ["x_m", "y_m", "z_m"]
    np.testing.assert_allclose(
        points[coordinate_columns].to_numpy(dtype=float),
        np.array(
            [
                [1.25, 2.5, 3.75],
                [-4.0, 5.5, 6.25],
            ]
        ),
    )
    assert all(points[column].dtype.isnative for column in coordinate_columns)
