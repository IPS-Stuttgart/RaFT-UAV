from __future__ import annotations

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


def test_vertical_repair_rejects_whitespace_equivalent_duplicate_keys() -> None:
    rows = _submission()
    rows.loc[1, "sequence_id"] = " seq0001 "
    rows.loc[1, "time_s"] = 0.0

    with pytest.raises(ValueError, match=r"seq0001@0"):
        repair_track5_vertical_spikes(rows)


def test_vertical_repair_canonicalizes_sequence_identifiers() -> None:
    rows = _submission()
    rows.loc[1, "sequence_id"] = " seq0001 "

    repaired, diagnostics = repair_track5_vertical_spikes(rows)

    assert repaired["sequence_id"].tolist() == ["seq0001"] * len(rows)
    assert set(diagnostics["sequence_id"]) == {"seq0001"}
