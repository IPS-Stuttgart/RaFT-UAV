from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from raft_uav.mmuad.pointcloud2 import pointcloud2_to_dataframe


def test_pointcloud2_decoder_keeps_field_reads_inside_each_record() -> None:
    record_width = 16
    fields = [
        SimpleNamespace(name="x", offset=0, datatype=7, count=1),
        SimpleNamespace(name="y", offset=record_width, datatype=7, count=1),
        SimpleNamespace(name="z", offset=8, datatype=7, count=1),
    ]
    message = SimpleNamespace(
        fields=fields,
        data=np.zeros(8, dtype="<f4").tobytes(),
        width=2,
        height=1,
        point_step=record_width,
        row_step=record_width * 2,
        is_bigendian=False,
    )

    decoded = pointcloud2_to_dataframe(message)

    assert decoded.empty
