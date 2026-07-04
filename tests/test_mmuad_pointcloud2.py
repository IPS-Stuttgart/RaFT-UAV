import struct
from types import SimpleNamespace
from typing import Any

import pytest

from raft_uav.mmuad.pointcloud2 import pointcloud2_to_candidates, pointcloud2_to_dataframe


_FLOAT32 = 7


def _field(name: object, offset: int) -> SimpleNamespace:
    return SimpleNamespace(name=name, offset=offset, datatype=_FLOAT32, count=1)


def _message(*fields: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        fields=list(fields),
        data=struct.pack("<fff", 1.0, 2.0, 3.0),
        width=1,
        height=1,
        point_step=12,
        row_step=12,
        is_bigendian=False,
    )


def _empty_message(*fields: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        fields=list(fields),
        data=b"",
        width=0,
        height=1,
        point_step=12,
        is_bigendian=False,
    )


def test_pointcloud2_decoder_accepts_bytes_and_padded_field_names():
    frame = pointcloud2_to_dataframe(
        _message(
            _field(b"x\x00", 0),
            _field(" y ", 4),
            _field(b" z ", 8),
        )
    )

    assert frame.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


def test_pointcloud2_decoder_strips_nul_padding_after_whitespace():
    frame = pointcloud2_to_dataframe(
        _message(
            _field(b" x\x00 ", 0),
            _field(" y\x00 ", 4),
            _field(b" z\x00 ", 8),
        )
    )

    assert frame.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


def test_pointcloud2_decoder_zero_height_uses_flat_complete_records() -> None:
    message = SimpleNamespace(
        fields=[_field("x", 0), _field("y", 4), _field("z", 8)],
        data=struct.pack("<ffffff", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        width=1,
        height=0,
        point_step=12,
        row_step=12,
        is_bigendian=False,
    )

    frame = pointcloud2_to_dataframe(message)

    assert frame[["x_m", "y_m", "z_m"]].values.tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]


@pytest.mark.parametrize(
    "time_s",
    [float("nan"), float("inf"), -float("inf"), True, False, "bad"],
)
def test_pointcloud2_to_candidates_rejects_malformed_timestamps(time_s: Any) -> None:
    with pytest.raises(ValueError, match="time_s must be a finite numeric timestamp"):
        pointcloud2_to_candidates(
            _empty_message(_field("x", 0), _field("y", 4), _field("z", 8)),
            sequence_id="seq0",
            time_s=time_s,
        )
