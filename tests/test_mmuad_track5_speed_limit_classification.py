from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 100.0, 200.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize("value", [-1, 4, 1.5])
def test_speed_limit_rejects_invalid_classification_values(value: object) -> None:
    rows = _submission()
    rows["Classification"] = rows["Classification"].astype(object)
    rows.loc[1, "Classification"] = value

    with pytest.raises(ValueError) as error:
        project_track5_speed_limit(rows)

    message = str(error.value)
    assert "submission contains invalid Classification values" in message
    assert "1:" in message


def test_speed_limit_canonicalizes_integer_equivalent_classification_values() -> None:
    rows = _submission()
    rows["Classification"] = ["0", "1.0", 2.0]

    limited, _ = project_track5_speed_limit(rows)

    assert limited["Classification"].tolist() == [0, 1, 2]
    assert limited["Classification"].dtype.kind in {"i", "u"}
