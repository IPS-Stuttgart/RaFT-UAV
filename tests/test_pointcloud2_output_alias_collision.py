"""Regression tests for PointCloud2 decoded-coordinate alias collisions."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import pytest

from raft_uav.mmuad.pointcloud2 import pointcloud2_to_dataframe


@dataclass(frozen=True)
class Field:
    name: object
    offset: int
    datatype: int = 7
    count: int = 1


@dataclass(frozen=True)
class Message:
    fields: list[Field]
    data: bytes
    width: int = 1
    height: int = 1
    point_step: int = 16
    row_step: int = 16
    is_bigendian: bool = False


@pytest.mark.parametrize(
    ("field_name", "normalized_name"),
    [
        ("x_m", "x_m"),
        (" Y_M\x00 ", "y_m"),
        (b" Z_M ", "z_m"),
    ],
)
def test_pointcloud2_rejects_decoded_coordinate_alias_fields(
    field_name: object,
    normalized_name: str,
) -> None:
    message = Message(
        fields=[
            Field("x", 0),
            Field("y", 4),
            Field("z", 8),
            Field(field_name, 12),
        ],
        data=struct.pack("<ffff", 1.0, 2.0, 3.0, 99.0),
    )

    with pytest.raises(
        ValueError,
        match=rf"field name '{normalized_name}' conflicts with decoded coordinate column",
    ):
        pointcloud2_to_dataframe(message)
