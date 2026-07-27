from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.io.aerpaw import select_radar_measurement_rows


def test_catprob_preserves_frames_with_partial_indices() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0.0, 0.0, 1.0, np.nan, np.nan, np.nan],
            "time_s": [0.0, 0.0, 0.0, 1.0, 2.0, 2.0],
            "track_id": [10, 12, 11, 20, 30, 31],
            "track_index": [0, 1, 0, 0, 0, 1],
            "cat_prob_uav": [0.90, 0.95, 0.80, 0.70, 0.60, 0.96],
        }
    )

    selected = select_radar_measurement_rows(
        radar,
        selection="catprob",
        catprob_threshold=0.5,
    )

    assert selected["track_id"].tolist() == [12, 11, 20, 31]
    assert selected["time_s"].tolist() == [0.0, 0.0, 1.0, 2.0]
