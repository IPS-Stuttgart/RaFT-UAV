from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import _normalized_submission


def _normalized_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 10.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("time_s", "invalid"),
        ("state_x_m", float("nan")),
    ],
)
def test_acceleration_normalizer_rejects_invalid_rows_instead_of_dropping_them(
    column: str,
    value: object,
) -> None:
    rows = _normalized_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value

    with pytest.raises(ValueError, match=r"row indices: 1$"):
        _normalized_submission(rows)
