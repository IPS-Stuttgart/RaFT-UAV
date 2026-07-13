"""Tests for ROS PointCloud2 decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
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


@pytest.mark.parametrize(
    ("malformed_field", "match"),
    [
        (Field("x", 0, datatype=99), "unsupported datatype 99"),
        (Field("y", 4, count=2), "must have count 1"),
        (Field("z", 12), "does not fit within point_step 12"),
    ],
)
def test_pointcloud2_rejects_malformed_required_xyz_fields(
    malformed_field: Field,
    match: str,
) -> None:
    base = _xyz_message(row_step=24)
    fields = [
        malformed_field if field.name == malformed_field.name else field
        for field in base.fields
    ]
    malformed = Message(
        fields=fields,
        data=base.data,
        width=base.width,
        height=base.height,
        point_step=base.point_step,
        row_step=base.row_step,
        is_bigendian=base.is_bigendian,
    )

    with pytest.raises(ValueError, match=match):
        pointcloud2_to_dataframe(malformed)
