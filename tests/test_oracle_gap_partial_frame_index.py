from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.oracle_gap_decomposition import (
    OracleGapConfig,
    decompose_radar_oracle_gap,
)


def test_oracle_gap_preserves_frames_with_missing_frame_indices() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "frame_index": [0.0, 0.0, np.nan, np.nan],
            "track_id": [10, 11, 20, 21],
            "east_m": [0.0, 20.0, 1.0, 21.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        config=OracleGapConfig(plausible_candidate_gate_m=5.0),
    )

    assert rows["frame_key_type"].tolist() == ["frame_index", "time_s"]
    assert rows["frame_key"].tolist() == [0, 1.0]
    assert rows["candidate_count"].tolist() == [2, 2]
    assert rows["nearest_candidate_track_id"].tolist() == [10, 20]
    assert rows["category"].tolist() == [
        "plausible_candidate_not_selected",
        "plausible_candidate_not_selected",
    ]
