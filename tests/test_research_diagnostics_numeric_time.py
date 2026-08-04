from __future__ import annotations

import pandas as pd

from raft_uav.research import track_switch_metrics as package_track_switch_metrics
from raft_uav.research.diagnostics import (
    track_switch_metrics as module_track_switch_metrics,
)


def test_track_switch_metrics_sorts_numeric_string_timestamps_numerically() -> None:
    """Public diagnostic imports must sort numeric strings chronologically."""
    numeric = pd.DataFrame(
        {
            "time_s": [1.0, 2.0, 10.0],
            "track_id": [7, 7, 9],
        }
    )
    strings = numeric.assign(time_s=numeric["time_s"].astype(str))

    for metric in (package_track_switch_metrics, module_track_switch_metrics):
        expected = metric(numeric)
        actual = metric(strings)

        assert actual["track_switch_count"] == expected["track_switch_count"] == 1
        assert actual["long_gap_count"] == expected["long_gap_count"] == 1
        assert actual["max_selected_gap_s"] == expected["max_selected_gap_s"] == 8.0
