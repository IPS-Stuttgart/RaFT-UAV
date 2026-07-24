from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import validate_mmaud_results_frame


def _result_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "timestamp": [0.0, 1.0],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
            "uav_type": ["0", "0"],
            "score": [1.0, 1.0],
        }
    )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("timestamp", "not-a-number"),
        ("x", np.nan),
        ("score", np.inf),
    ],
)
def test_evaluator_rejects_mixed_invalid_trajectory_rows(
    column: str,
    invalid_value: object,
) -> None:
    rows = _result_rows()
    rows.loc[1, column] = invalid_value

    with pytest.raises(
        ValueError,
        match="1 non-finite or non-numeric trajectory row",
    ):
        validate_mmaud_results_frame(rows)
