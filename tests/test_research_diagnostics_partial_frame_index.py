from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.diagnostics import association_regret
from raft_uav.research.diagnostics import candidate_set_recall


def _partial_index_radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [0.0, np.nan, 2.0],
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 10.0, 20.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [5.0, 5.0, 5.0],
            "track_id": [11, 12, 13],
        }
    )


def _three_frame_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 10.0, 20.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [5.0, 5.0, 5.0],
        }
    )


def test_research_diagnostics_preserve_partially_indexed_radar_frames() -> None:
    radar = _partial_index_radar()
    truth = _three_frame_truth()

    recall = candidate_set_recall(
        radar,
        truth,
        distance_gate_m=0.1,
        max_time_delta_s=0.1,
    )
    regret = association_regret(
        radar,
        radar,
        truth,
        max_time_delta_s=0.1,
    )

    assert recall["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert recall["target_present"].tolist() == [True, True, True]
    np.testing.assert_allclose(recall["best_candidate_error_m"], 0.0)
    assert regret["candidate_count"].tolist() == [1, 1, 1]
    np.testing.assert_allclose(regret["selected_error_m"], 0.0)
    np.testing.assert_allclose(regret["best_candidate_error_m"], 0.0)
    np.testing.assert_allclose(regret["association_regret_m"], 0.0)
