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


def test_speed_limit_rejects_whitespace_equivalent_duplicate_keys() -> None:
    rows = _submission()
    rows.loc[1, "sequence_id"] = " seq0001 "
    rows.loc[1, "time_s"] = 0.0

    with pytest.raises(ValueError, match=r"seq0001@0"):
        project_track5_speed_limit(rows)


def test_speed_limit_canonicalizes_sequence_identifiers() -> None:
    rows = _submission()
    rows.loc[1, "sequence_id"] = " seq0001 "

    limited, diagnostics = project_track5_speed_limit(rows)

    assert limited["sequence_id"].tolist() == ["seq0001"] * len(rows)
    assert diagnostics["sequence_id"].tolist() == ["seq0001"] * len(rows)
