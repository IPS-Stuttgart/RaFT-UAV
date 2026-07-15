from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.paper_selection import (
    range_gated_radar_candidates,
    select_paper_strict_raw_radar_track,
)


def test_catprob_gate_excludes_nonfinite_values() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "track_id": [1, 1, 1],
            "range_m": [100.0, 100.0, 100.0],
            "cat_prob_uav": [np.inf, np.nan, 0.8],
        }
    )

    selected = range_gated_radar_candidates(
        radar,
        catprob_threshold=0.5,
        require_range_m=False,
    )

    assert selected["time_s"].tolist() == [2.0]
    assert selected["association_catprob_candidate_rows"].tolist() == [3]


def test_nonfinite_catprob_does_not_win_track_tie_break() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "track_id": [1, 1, 2, 2],
            "cat_prob_uav": [0.9, 0.9, np.inf, 0.1],
        }
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["track_id"].unique().tolist() == [1]
