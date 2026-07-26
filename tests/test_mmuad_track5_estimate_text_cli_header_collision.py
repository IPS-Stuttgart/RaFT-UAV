from __future__ import annotations

from io import StringIO

import pytest

from raft_uav.mmuad.track5_estimate_text_cli import _read_csv_preserving_sequence_id


def test_estimate_fit_wrapper_rejects_headers_colliding_after_strip() -> None:
    csv_stream = StringIO(
        "sequence_id, sequence_id ,time_s,state_x_m,state_y_m,state_z_m\n"
        "001,002,0.0,1.0,2.0,3.0\n"
    )

    with pytest.raises(
        ValueError,
        match=r"ambiguous columns after trimming whitespace.*sequence_id",
    ):
        _read_csv_preserving_sequence_id(csv_stream)
