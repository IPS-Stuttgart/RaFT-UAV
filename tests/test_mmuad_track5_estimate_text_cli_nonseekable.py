from __future__ import annotations

from io import StringIO, UnsupportedOperation

from raft_uav.mmuad.track5_estimate_text_cli import _read_csv_preserving_sequence_id


class _NonSeekableTextStream:
    def __init__(self, text: str) -> None:
        self._stream = StringIO(text)

    def read(self, *args, **kwargs):
        return self._stream.read(*args, **kwargs)

    def readline(self, *args, **kwargs):
        return self._stream.readline(*args, **kwargs)

    def tell(self) -> int:
        return self._stream.tell()

    def seek(self, *_args, **_kwargs):
        raise UnsupportedOperation("stream is not seekable")


def test_estimate_fit_wrapper_does_not_consume_nonseekable_stream() -> None:
    csv_stream = _NonSeekableTextStream(
        "sequence_id,time_s,state_x_m,state_y_m,state_z_m\n"
        "001,0.0,1.0,2.0,3.0\n"
    )

    rows = _read_csv_preserving_sequence_id(csv_stream)

    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "time_s"] == 0.0
