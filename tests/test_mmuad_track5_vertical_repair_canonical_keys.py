from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def test_vertical_repair_rejects_duplicate_keys_after_sequence_normalization() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", " seq0001 "],
            "time_s": [0.0, 1.0, "1"],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 100.0, 2.0],
            "Classification": [2, 2, 2],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"1 duplicate \(sequence_id, time_s\) key\(s\): seq0001@1$",
    ):
        repair_track5_vertical_spikes(rows)
