"""Regression tests for strict PointCloud2 integer metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
import struct
from typing import Any

import pytest

from raft_uav.mmuad.pointcloud2 import pointcloud2_to_dataframe


@dataclass(frozen=True)
class Field:
    name: str
    offset: Any
    datatype: Any = 7
    count: Any = 1


@dataclass(frozen=True)
class Message:
    fields: list[Field]
    data: bytes
    width: Any
    height: Any
    point_step: Any
    row_step: Any = None
    is_bigendian: bool = False


def _message() -> Message:
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    return Message(
        fields=[Field("x", 0), Field("y", 4), Field("z", 8)],
        data=b"".join(struct.pack("<fff", *point) for point in points),
        width=2,
        height=1,
        point_step=12,
        row_step=24,
    )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("offset", 0.5),
        ("datatype", 7.5),
        ("count", 1.5),
        ("offset", True),
    ],
)
def test_pointcloud2_rejects_lossy_field_integer_coercion(
    attribute: str,
    value: Any,
) -> None:
    message = _message()
    malformed_x = replace(message.fields[0], **{attribute: value})
    malformed = replace(message, fields=[malformed_x, *message.fields[1:]])

    with pytest.raises(ValueError, match=rf"field 'x' {attribute} must be an integer scalar"):
        pointcloud2_to_dataframe(malformed)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("point_step", 12.5),
        ("width", 2.5),
        ("height", 1.5),
        ("row_step", 24.5),
        ("width", False),
    ],
)
def test_pointcloud2_rejects_lossy_layout_integer_coercion(
    attribute: str,
    value: Any,
) -> None:
    malformed = replace(_message(), **{attribute: value})

    with pytest.raises(ValueError, match=rf"{attribute} must be an integer scalar"):
        pointcloud2_to_dataframe(malformed)
