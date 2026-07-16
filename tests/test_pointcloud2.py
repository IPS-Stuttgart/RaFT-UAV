"""Tests for ROS PointCloud2 decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import struct

import pytest

from raft_uav.mmuad.pointcloud2 import pointcloud2_to_dataframe


@dataclass(frozen=True)
class Field:
    name: str
    offset: int
    datatype: int = 7
    count: int = 1


@dataclass(frozen=True)
class Message:
    fields: list[Field]
    data: bytes
    width: int
    height: int
    point_step: int
    row_step: int | None = None
    is_bigendian: bool = False


def _xyz_message(*, row_step: int | None) -> Message:
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    data = b"".join(struct.pack("<fff", *point) for point in points)
    return Message(
        fields=[Field("x", 0), Field("y", 4), Field("z", 8)],
        data=data,
        width=2,
        height=1,
        point_step=12,
        row_step=row_step,
    )


def test_pointcloud2_decodes_contiguous_export_with_zero_row_step() -> None:
    frame = pointcloud2_to_dataframe(_xyz_message(row_step=0))

    assert frame[["x_m", "y_m", "z_m"]].to_dict("records") == [
        {"x_m": 1.0, "y_m": 2.0, "z_m": 3.0},
        {"x_m": 4.0, "y_m": 5.0, "z_m": 6.0},
    ]


def test_pointcloud2_rejects_negative_row_step() -> None:
    with pytest.raises(ValueError, match="row_step must be positive"):
        pointcloud2_to_dataframe(_xyz_message(row_step=-12))


def test_pointcloud2_rejects_truncated_organized_cloud() -> None:
    message = replace(_xyz_message(row_step=0), height=2)

    with pytest.raises(ValueError, match=r"data is shorter than height \* row_step"):
        pointcloud2_to_dataframe(message)
