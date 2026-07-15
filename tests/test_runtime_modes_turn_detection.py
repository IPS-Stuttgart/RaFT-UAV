from __future__ import annotations

import pandas as pd

from raft_uav.research.runtime_modes import segment_flight_phases


def test_segment_flight_phases_detects_direction_change() -> None:
    frame = pd.DataFrame(
        {
            "time_s": range(8),
            "east_m": [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "north_m": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "up_m": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )

    phases = segment_flight_phases(frame)

    assert phases.iloc[4] == "turn"
    assert phases.tolist().count("turn") == 1


def test_segment_flight_phases_does_not_invent_turn_at_invalid_step() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 1.0, 2.0, 3.0],
            "east_m": [0.0, 1.0, 2.0, 3.0, 4.0],
            "north_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    phases = segment_flight_phases(frame)

    assert "turn" not in phases.tolist()
