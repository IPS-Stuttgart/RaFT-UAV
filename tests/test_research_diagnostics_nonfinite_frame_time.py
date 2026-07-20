from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.diagnostics import association_regret, candidate_set_recall


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def _radar_with_nonfinite_time() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [7],
            "time_s": [np.nan],
            "east_m": [10.0],
            "north_m": [0.0],
            "up_m": [0.0],
            "track_id": [1],
        }
    )


def test_candidate_recall_does_not_match_nonfinite_frame_time() -> None:
    recall = candidate_set_recall(
        _radar_with_nonfinite_time(),
        _truth(),
        distance_gate_m=0.1,
    )

    assert len(recall) == 1
    assert np.isnan(recall.loc[0, "truth_time_delta_s"])
    assert np.isnan(recall.loc[0, "best_candidate_error_m"])
    assert not bool(recall.loc[0, "target_present"])


def test_association_regret_does_not_score_nonfinite_selected_time() -> None:
    radar = _radar_with_nonfinite_time()

    regret = association_regret(radar, radar, _truth())

    assert len(regret) == 1
    assert regret.loc[0, "candidate_count"] == 1
    assert np.isnan(regret.loc[0, "truth_time_delta_s"])
    assert np.isnan(regret.loc[0, "selected_error_m"])
    assert np.isnan(regret.loc[0, "best_candidate_error_m"])
    assert np.isnan(regret.loc[0, "association_regret_m"])
