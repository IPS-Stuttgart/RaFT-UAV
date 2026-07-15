import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.time_offset import select_radar_rows_for_offset


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [10.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


@pytest.mark.parametrize(
    "selection",
    ["oracle-nearest-truth", "catprob-oracle-nearest", "highest-catprob"],
)
def test_time_offset_selection_skips_malformed_radar_positions(selection):
    group = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_id": [1, 2],
            "cat_prob_uav": [0.99, 0.5],
            "east_m": ["malformed", 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    selected_times, selected_positions = select_radar_rows_for_offset(
        groups=[group],
        truth=_truth(),
        tau_s=0.0,
        selection=selection,
        catprob_threshold=0.4,
        longest_track_id=None,
        max_truth_time_delta_s=1.0,
    )

    assert selected_times == [0.0]
    assert len(selected_positions) == 1
    assert np.allclose(selected_positions[0], np.array([10.0, 0.0, 0.0]))


def test_time_offset_selection_skips_frames_without_finite_positions():
    group = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "cat_prob_uav": [0.9, 0.8],
            "east_m": ["malformed", np.inf],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    selected_times, selected_positions = select_radar_rows_for_offset(
        groups=[group],
        truth=_truth(),
        tau_s=0.0,
        selection="highest-catprob",
        catprob_threshold=0.4,
        longest_track_id=None,
        max_truth_time_delta_s=1.0,
    )

    assert selected_times == []
    assert selected_positions == []
