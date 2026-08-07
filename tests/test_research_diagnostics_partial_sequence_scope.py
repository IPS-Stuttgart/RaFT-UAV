from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research import track_switch_metrics as package_track_switch_metrics
from raft_uav.research.diagnostics import (
    track_switch_metrics as module_track_switch_metrics,
)


TRACK_SWITCH_METRICS = (
    package_track_switch_metrics,
    module_track_switch_metrics,
)


def _partially_labeled_selection(track_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["flight-a", "flight-a", None, None],
            "time_s": [0.0, 1.0, 100.0, 101.0],
            "track_id": track_ids,
        }
    )


def test_track_switch_metrics_isolate_missing_rows_from_explicit_sequence() -> None:
    selected = _partially_labeled_selection([10, 10, 20, 20])

    for metric in TRACK_SWITCH_METRICS:
        result = metric(selected, long_gap_s=5.0)

        assert result["selected_radar_rows"] == 4
        assert result["track_switch_count"] == 0
        assert result["long_gap_count"] == 0
        assert result["max_selected_gap_s"] == 1.0


def test_track_switch_metrics_keep_reused_ids_distinct_from_missing_group() -> None:
    selected = _partially_labeled_selection([7, 7, 7, 7])

    for metric in TRACK_SWITCH_METRICS:
        result = metric(selected, long_gap_s=5.0)

        assert result["unique_track_ids"] == 2
        assert np.isclose(result["dominant_track_fraction"], 0.5)
        assert np.isclose(result["track_id_entropy"], 1.0)
        assert result["long_gap_count"] == 0
        assert result["max_selected_gap_s"] == 1.0
