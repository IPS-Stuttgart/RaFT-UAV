from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.factor_graph import _initial_radar_selection


def test_initial_selection_ignores_higher_scoring_nonfinite_position() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [1.0, 1.0],
            "frame_index": [4, 4],
            "track_id": [10, 11],
            "east_m": [np.nan, 10.0],
            "north_m": [np.nan, 20.0],
            "up_m": [np.nan, 30.0],
            "cat_prob_uav": [0.99, 0.80],
        }
    )

    selected = _initial_radar_selection(radar)

    assert selected["track_id"].tolist() == [11]


def test_initial_selection_falls_back_when_scores_are_nonfinite() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [1.0, 1.0],
            "frame_index": [4, 4],
            "track_id": [10, 11],
            "east_m": [10.0, 11.0],
            "north_m": [20.0, 21.0],
            "up_m": [30.0, 31.0],
            "cat_prob_uav": [np.nan, np.inf],
        }
    )

    selected = _initial_radar_selection(radar)

    assert selected["track_id"].tolist() == [10]


def test_initial_selection_skips_frames_without_finite_positions() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [1.0, 2.0],
            "frame_index": [4, 5],
            "track_id": [10, 20],
            "east_m": [np.nan, 10.0],
            "north_m": [np.nan, 20.0],
            "up_m": [np.nan, 30.0],
            "cat_prob_uav": [0.99, 0.80],
        }
    )

    selected = _initial_radar_selection(radar)

    assert selected["track_id"].tolist() == [20]
