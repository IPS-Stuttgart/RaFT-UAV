from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.pyrecest_tracklet_graph import fortem_tracklet_from_summary


def test_summary_adapter_falls_back_across_pandas_missing_aliases() -> None:
    row = pd.Series(
        {
            "segment_id": pd.NA,
            "micro_segment_id": "segment-7",
            "track_id": pd.NA,
            "start_time_s": pd.NA,
            "time_start_s": 10.0,
            "end_time_s": pd.NA,
            "time_end_s": 12.0,
            "start_east_m": pd.NA,
            "first_east_m": 1.0,
            "start_north_m": pd.NA,
            "first_north_m": 2.0,
            "start_up_m": pd.NA,
            "first_up_m": 3.0,
            "end_east_m": pd.NA,
            "last_east_m": 4.0,
            "end_north_m": pd.NA,
            "last_north_m": 5.0,
            "end_up_m": pd.NA,
            "last_up_m": 6.0,
            "score": pd.NA,
        }
    )

    tracklet = fortem_tracklet_from_summary(row)

    assert tracklet.id == "segment-7"
    assert tracklet.start_time == 10.0
    assert tracklet.end_time == 12.0
    np.testing.assert_allclose(tracklet.start_state, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(tracklet.end_state, [4.0, 5.0, 6.0])
    assert tracklet.cost == 0.0
    assert tracklet.metadata is not None
    assert tracklet.metadata["track_id"] is None
    assert tracklet.metadata["score"] is None
