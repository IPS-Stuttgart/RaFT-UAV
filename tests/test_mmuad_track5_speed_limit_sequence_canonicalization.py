from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _submission(sequence_ids: list[str], time_s: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "time_s": time_s,
            "state_x_m": [0.0, 100.0, 200.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


def test_speed_limit_rejects_whitespace_equivalent_duplicate_keys() -> None:
    rows = _submission(
        ["seq0001", " seq0001 ", "seq0001"],
        [0.0, 1.0, 1.0],
    )

    with pytest.raises(
        ValueError,
        match=r"1 duplicate \(sequence_id, time_s\) key\(s\): seq0001@1$",
    ):
        project_track5_speed_limit(rows, max_speed_mps=10.0)


def test_speed_limit_canonicalizes_sequence_identifiers() -> None:
    rows = _submission(
        [" seq0001 ", "seq0001", " seq0001"],
        [0.0, 1.0, 2.0],
    )

    limited, diagnostics = project_track5_speed_limit(rows, max_speed_mps=10.0)

    assert limited["sequence_id"].tolist() == ["seq0001", "seq0001", "seq0001"]
    assert diagnostics["sequence_id"].tolist() == ["seq0001", "seq0001", "seq0001"]
