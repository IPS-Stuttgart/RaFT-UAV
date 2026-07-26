from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import repair_track5_acceleration_kinks


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize("value", [-1, 4, 1.5, "not-a-class", None])
def test_acceleration_limit_rejects_invalid_classification_values(value: object) -> None:
    rows = _submission()
    rows["Classification"] = rows["Classification"].astype(object)
    rows.loc[1, "Classification"] = value

    with pytest.raises(
        ValueError,
        match=r"invalid Classification values at rows 1:",
    ):
        repair_track5_acceleration_kinks(rows)


def test_acceleration_limit_accepts_integer_equivalent_classification_values() -> None:
    rows = _submission()
    rows["Classification"] = ["0", "1.0", 2.0]

    repaired, _ = repair_track5_acceleration_kinks(rows)

    assert repaired["Classification"].tolist() == [0.0, 1.0, 2.0]
