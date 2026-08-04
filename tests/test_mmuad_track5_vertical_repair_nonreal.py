from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_y_m": [0.0] * 5,
            "state_z_m": [10.0, 11.0, 80.0, 13.0, 14.0],
            "Classification": [2] * 5,
        }
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("state_z_m", 1.0 + 2.0j),
        ("state_x_m", np.array(1.0 + 2.0j)),
        (
            "state_y_m",
            np.array(np.array(1.0 + 2.0j), dtype=object),
        ),
    ],
)
def test_vertical_repair_rejects_complex_numeric_cells(
    column: str,
    value: object,
) -> None:
    rows = _submission_rows()
    rows[column] = rows[column].astype(object)
    rows.at[2, column] = value

    with pytest.raises(
        ValueError,
        match=rf"complex numeric values: {column} rows \[2\]",
    ):
        repair_track5_vertical_spikes(rows)


def test_vertical_repair_rejects_wrapped_boolean_numeric_cells() -> None:
    rows = _submission_rows()
    rows["time_s"] = rows["time_s"].astype(object)
    rows.at[2, "time_s"] = np.array(True, dtype=object)

    with pytest.raises(
        ValueError,
        match=r"Boolean numeric values: time_s rows \[2\]",
    ):
        repair_track5_vertical_spikes(rows)


def test_vertical_repair_rejects_wrapped_boolean_controls() -> None:
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        repair_track5_vertical_spikes(
            _submission_rows(),
            iterations=np.array(True, dtype=object),
        )
