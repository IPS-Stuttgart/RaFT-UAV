from __future__ import annotations

import numpy as np
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


@pytest.mark.parametrize("value", [None, pd.NA, np.nan, "", "   "])
def test_speed_limit_rejects_missing_or_blank_sequence_ids(value: object) -> None:
    rows = _submission()
    rows.loc[1, "sequence_id"] = value

    with pytest.raises(ValueError) as error:
        project_track5_speed_limit(rows)

    message = str(error.value)
    assert "submission contains missing or blank sequence_id values" in message
    assert "sequence_id rows [1]" in message


def test_speed_limit_keeps_nonblank_numeric_sequence_ids() -> None:
    rows = _submission()
    rows["sequence_id"] = [0, 0, 0]

    limited, diagnostics = project_track5_speed_limit(
        rows,
        max_speed_mps=10.0,
        iterations=1,
    )

    assert limited["sequence_id"].tolist() == ["0", "0", "0"]
    assert diagnostics["sequence_id"].tolist() == ["0", "0", "0"]
