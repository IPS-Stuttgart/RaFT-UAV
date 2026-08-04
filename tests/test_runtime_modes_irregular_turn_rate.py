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


def test_segment_flight_phases_sorts_numeric_string_timestamps_numerically() -> None:
    numeric_frame = pd.DataFrame(
        {
            "time_s": [1.0, 2.0, 10.0, 11.0, 12.0, 13.0],
            "east_m": [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
            "north_m": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            "up_m": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    string_frame = numeric_frame.assign(time_s=numeric_frame["time_s"].astype(str))

    expected = segment_flight_phases(numeric_frame)
    actual = segment_flight_phases(string_frame)

    pd.testing.assert_series_equal(actual, expected)
