from __future__ import annotations

import pandas as pd

from raft_uav.research.runtime_modes import segment_flight_phases


def test_segment_flight_phases_uses_angular_rate_for_irregular_samples() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "east_m": [0.0, 1.0, 2.0, 2.0, 1.0, 0.0, -1.0, -2.0],
            "north_m": [0.0, 0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "up_m": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )

    phases = segment_flight_phases(frame)

    assert phases.iloc[3] != "turn"
    assert phases.iloc[4] == "turn"
    assert phases.tolist().count("turn") == 1
