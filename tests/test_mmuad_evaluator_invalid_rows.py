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
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = invalid_value

    with pytest.raises(
        ValueError,
        match="1 non-finite or non-numeric trajectory row",
    ):
        validate_mmaud_results_frame(rows)


@pytest.mark.parametrize("column", ["timestamp", "x", "y", "z", "score"])
@pytest.mark.parametrize("value", [1.0 + 2.0j, np.complex128(1.0 + 0.0j)])
def test_evaluator_rejects_complex_trajectory_rows(
    column: str,
    value: complex,
) -> None:
    rows = _result_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value

    with pytest.raises(ValueError, match="1 complex trajectory row"):
        validate_mmaud_results_frame(rows)


def test_evaluator_rejects_complex_trajectory_aliases() -> None:
    rows = _result_rows().rename(columns={"timestamp": "time_s", "score": "confidence"})
    rows["confidence"] = rows["confidence"].astype(object)
    rows.loc[1, "confidence"] = 0.5 + 0.25j

    with pytest.raises(ValueError, match="1 complex trajectory row"):
        validate_mmaud_results_frame(rows)
