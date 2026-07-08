from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.evaluator import validate_mmaud_results_frame


def test_validate_mmaud_results_accepts_padded_official_track5_headers() -> None:
    frame = pd.DataFrame(
        {
            " Sequence ": ["seq0001"],
            " TIMESTAMP ": [1.25],
            " Position ": ["(1.0, 2.0, 3.0)"],
            " Classification ": [2],
        }
    )

    rows = validate_mmaud_results_frame(frame)

    assert list(rows.columns) == [
        "sequence_id",
        "timestamp",
        "x",
        "y",
        "z",
        "uav_type",
        "score",
    ]
    assert rows.loc[0, "sequence_id"] == "seq0001"
    assert rows.loc[0, "timestamp"] == 1.25
    assert rows.loc[0, "x"] == 1.0
    assert rows.loc[0, "y"] == 2.0
    assert rows.loc[0, "z"] == 3.0
    assert rows.loc[0, "uav_type"] == "2"
    assert rows.loc[0, "score"] == 1.0
