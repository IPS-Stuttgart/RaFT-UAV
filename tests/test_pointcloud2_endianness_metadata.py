"""Regression tests for strict PointCloud2 endianness metadata."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

import numpy as np
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
    width: int = 1
    height: int = 1
    point_step: int = 12
    row_step: int = 12
    is_bigendian: Any = False


def _message(*, byte_order: str, is_bigendian: Any) -> Message:
    return Message(
        fields=[Field("x", 0), Field("y", 4), Field("z", 8)],
        data=struct.pack(f"{byte_order}fff", 1.0, 2.0, 3.0),
        is_bigendian=is_bigendian,
    )


def test_pointcloud2_serialized_false_keeps_little_endian_decoding() -> None:
    frame = pointcloud2_to_dataframe(
        _message(byte_order="<", is_bigendian="false")
    )

    assert frame.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


def test_pointcloud2_serialized_true_keeps_big_endian_decoding() -> None:
    frame = pointcloud2_to_dataframe(
        _message(byte_order=">", is_bigendian="true")
    )

    assert frame.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "is_bigendian",
    [2, -1, 0.5, "maybe", np.array([False]), np.ma.masked],
)
def test_pointcloud2_rejects_malformed_endianness_metadata(
    is_bigendian: Any,
) -> None:
    with pytest.raises(ValueError, match="is_bigendian must be a Boolean scalar"):
        pointcloud2_to_dataframe(
            _message(byte_order="<", is_bigendian=is_bigendian)
        )
