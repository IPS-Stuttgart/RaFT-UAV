from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.io import _pcd_numpy_dtype, load_point_cloud_file_as_points


@pytest.mark.parametrize(
    ("size", "type_code"),
    [
        (2, "F"),
        (3, "I"),
        (16, "U"),
    ],
)
def test_pcd_dtype_rejects_unsupported_field_widths(size: int, type_code: str) -> None:
    with pytest.raises(ValueError, match=r"unsupported PCD SIZE"):
        _pcd_numpy_dtype(size=size, type_code=type_code)


def test_binary_pcd_loader_rejects_unsupported_float_width(tmp_path: Path) -> None:
    path = tmp_path / "invalid-size.pcd"
    header = "\n".join(
        [
            "VERSION .7",
            "FIELDS x y z",
            "SIZE 2 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            "WIDTH 1",
            "HEIGHT 1",
            "POINTS 1",
            "DATA binary",
            "",
        ]
    ).encode("ascii")
    path.write_bytes(header + bytes(16))

    with pytest.raises(
        ValueError,
        match=r"unsupported PCD SIZE 2 for TYPE 'F'",
    ):
        load_point_cloud_file_as_points(path)
