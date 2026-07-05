from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    normalize_official_track5_results_frame,
    parse_official_sequence_cell,
)


def test_parse_official_sequence_cell_accepts_numpy_integer_scalars() -> None:
    assert parse_official_sequence_cell(np.int64(7)) == "7"


def test_official_results_normalizer_accepts_numeric_sequence_column() -> None:
    frame = pd.DataFrame(
        {
            "Sequence": pd.Series([np.int64(7)], dtype="int64"),
            "Timestamp": [0.0],
            "Position": ["(1,2,3)"],
            "Classification": [1],
        }
    )

    normalized = normalize_official_track5_results_frame(frame)

    assert normalized.to_dict("records") == [
        {
            "Sequence": "7",
            "Timestamp": 0.0,
            "Position": "(1,2,3)",
            "Classification": 1,
        }
    ]
