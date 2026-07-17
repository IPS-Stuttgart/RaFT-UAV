from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_hampel_repair import repair_track5_hampel_spikes


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [5.0, 5.0, 5.0],
            "Classification": pd.Series([2, 2, 2], dtype=object),
        }
    )


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", np.nan, pd.NA, True, np.bool_(False)],
)
def test_hampel_repair_rejects_invalid_sequence_identifiers(value: object) -> None:
    rows = _submission_rows()
    rows.loc[1, "sequence_id"] = value

    with pytest.raises(ValueError, match=r"invalid sequence identifiers.*row positions: 1"):
        repair_track5_hampel_spikes(rows)


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf, -1, 4, 1.5, True, np.bool_(False), "false", "unknown"],
)
def test_hampel_repair_rejects_invalid_classification_values(value: object) -> None:
    rows = _submission_rows()
    rows.loc[1, "Classification"] = value

    with pytest.raises(ValueError, match=r"invalid Classification values.*row positions: 1"):
        repair_track5_hampel_spikes(rows)


def test_hampel_repair_preserves_valid_fixed_grid_identifiers() -> None:
    rows = _submission_rows()
    rows["Classification"] = ["2", 2.0, np.int64(2)]

    repaired, diagnostics = repair_track5_hampel_spikes(rows)

    assert repaired["sequence_id"].tolist() == ["seq0001", "seq0001", "seq0001"]
    assert repaired["Classification"].tolist() == [2, 2, 2]
    assert len(repaired) == len(rows)
    assert len(diagnostics) == len(rows)
