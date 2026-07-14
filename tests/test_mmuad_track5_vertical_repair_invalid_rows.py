from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 100.0, 2.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("time_s", np.nan),
        ("state_x_m", np.inf),
        ("state_y_m", -np.inf),
        ("state_z_m", "not-a-number"),
        ("Classification", pd.NA),
    ],
)
def test_vertical_repair_rejects_rows_with_invalid_numeric_values(
    column: str,
    value: object,
) -> None:
    rows = _submission()
    rows.loc[1, column] = value

    with pytest.raises(ValueError) as error:
        repair_track5_vertical_spikes(rows)

    assert "submission contains non-finite numeric values" in str(error.value)
    assert f"{column} rows [1]" in str(error.value)


def test_vertical_repair_reports_every_invalid_numeric_column() -> None:
    rows = _submission()
    rows.loc[0, "time_s"] = np.nan
    rows.loc[2, "state_z_m"] = np.inf

    with pytest.raises(ValueError) as error:
        repair_track5_vertical_spikes(rows)

    message = str(error.value)
    assert "time_s rows [0]" in message
    assert "state_z_m rows [2]" in message
