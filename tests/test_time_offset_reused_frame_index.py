from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.diagnostics.time_offset import (
    radar_frame_groups,
    sweep_radar_against_truth,
)


def _radar_with_reused_frame_counter() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "frame_index": [7, 7],
            "track_id": [101, 202],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.8, 0.9],
        }
    )


def test_radar_frame_groups_separates_reused_counters_by_timestamp() -> None:
    groups = radar_frame_groups(_radar_with_reused_frame_counter())

    assert [len(group) for group in groups] == [1, 1]
    assert [float(group["time_s"].iloc[0]) for group in groups] == [0.0, 10.0]


def test_time_offset_sweep_keeps_every_reused_counter_frame() -> None:
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    sweep = sweep_radar_against_truth(
        radar=_radar_with_reused_frame_counter(),
        truth=truth,
        taus_s=[0.0],
        dimensions=3,
        selection="highest-catprob",
        catprob_threshold=0.4,
        max_truth_time_delta_s=10.0,
    )

    result = sweep.iloc[0]
    assert int(result["candidate_count"]) == 2
    assert int(result["selected_count"]) == 2
    assert int(result["matched_count"]) == 2
    assert float(result["mean_error_m"]) == pytest.approx(0.0)
