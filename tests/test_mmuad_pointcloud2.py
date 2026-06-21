import struct
from types import SimpleNamespace

from raft_uav.mmuad.pointcloud2 import pointcloud2_to_dataframe


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


def test_pointcloud2_decoder_accepts_bytes_and_padded_field_names():
    frame = pointcloud2_to_dataframe(
        _message(
            _field(b"x\x00", 0),
            _field(" y ", 4),
            _field(b" z ", 8),
        )
    )

    assert frame.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]
