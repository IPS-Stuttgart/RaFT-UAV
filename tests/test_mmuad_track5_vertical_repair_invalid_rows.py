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
    rows[column] = rows[column].astype(object)
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


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("time_s", True),
        ("state_z_m", np.bool_(False)),
        ("Classification", True),
    ],
)
def test_vertical_repair_rejects_boolean_numeric_rows(
    column: str,
    value: object,
) -> None:
    rows = _submission()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value

    with pytest.raises(ValueError) as error:
        repair_track5_vertical_spikes(rows)

    message = str(error.value)
    assert "submission contains Boolean numeric values" in message
    assert f"{column} rows [1]" in message


def test_vertical_repair_rejects_duplicate_fixed_grid_keys() -> None:
    rows = _submission()
    rows.loc[2, "time_s"] = 1.0
    rows.loc[2, "state_z_m"] = 200.0

    with pytest.raises(ValueError) as error:
        repair_track5_vertical_spikes(rows)

    message = str(error.value)
    assert "1 duplicate (sequence_id, time_s) key(s)" in message
    assert "seq0001@1" in message


def test_vertical_repair_rejects_numeric_equivalent_duplicate_timestamps() -> None:
    rows = _submission()
    rows["time_s"] = rows["time_s"].astype(object)
    rows.loc[2, "time_s"] = "1"

    with pytest.raises(ValueError, match=r"seq0001@1"):
        repair_track5_vertical_spikes(rows)


def test_vertical_repair_allows_timestamp_reuse_across_sequences() -> None:
    rows = _submission()
    rows.loc[2, "sequence_id"] = "seq0002"
    rows.loc[2, "time_s"] = 1.0

    repaired, _ = repair_track5_vertical_spikes(rows)

    assert len(repaired) == len(rows)
    assert set(repaired["sequence_id"]) == {"seq0001", "seq0002"}
