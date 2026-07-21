from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.oracle_gap_decomposition import (
    OracleGapConfig,
    decompose_radar_oracle_gap,
)


def test_oracle_gap_does_not_match_nonfinite_radar_frame_time() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [np.nan],
            "frame_index": [7],
            "track_id": [11],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        config=OracleGapConfig(plausible_candidate_gate_m=10.0),
    )

    assert len(rows) == 1
    assert rows.loc[0, "frame_key"] == 7
    assert not bool(rows.loc[0, "truth_available"])
    assert np.isnan(rows.loc[0, "nearest_candidate_error_m"])
    assert not bool(rows.loc[0, "has_plausible_candidate"])
    assert rows.loc[0, "category"] == "no_truth"
