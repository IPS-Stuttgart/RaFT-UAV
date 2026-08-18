"""Regression tests for malformed PointCloud2 dimensions."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from raft_uav.mmuad.pointcloud2 import pointcloud2_to_dataframe


def _xyz_message(*, width: int, height: int) -> SimpleNamespace:
    fields = [
        SimpleNamespace(name="x", offset=0, datatype=7, count=1),
        SimpleNamespace(name="y", offset=4, datatype=7, count=1),
        SimpleNamespace(name="z", offset=8, datatype=7, count=1),
    ]
    return SimpleNamespace(
        fields=fields,
        data=struct.pack("<ffffff", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        width=width,
        height=height,
        point_step=12,
        row_step=12,
        is_bigendian=False,
    )


@pytest.mark.parametrize(
    ("width", "height", "dimension"),
    [
        (-1, 1, "width"),
        (1, -1, "height"),
    ],
)
def test_pointcloud2_rejects_negative_dimensions(
    width: int,
    height: int,
    dimension: str,
) -> None:
    message = _xyz_message(width=width, height=height)

    with pytest.raises(ValueError, match=rf"{dimension} must be non-negative"):
        pointcloud2_to_dataframe(message)
