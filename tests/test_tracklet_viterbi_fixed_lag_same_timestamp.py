import numpy as np
import pandas as pd

import raft_uav.baselines.tracklet_viterbi_fixed_lag as fixed_lag
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig


def _radar_event(frame_index: int) -> dict[str, object]:
    return {
        "kind": "radar",
        "time_s": 1.0,
        "candidates": pd.DataFrame(
            [
                {
                    "frame_index": frame_index,
                    "track_id": 10 + frame_index,
                    "time_s": 1.0,
                    "east_m": float(frame_index),
                    "north_m": 0.0,
                    "up_m": 0.0,
                    "cat_prob_uav": 1.0,
                }
            ]
        ),
    }


def test_fixed_lag_does_not_reprocess_committed_equal_time_frame(monkeypatch):
    radar_frame_ids_by_call: list[list[int]] = []

    def select_first_candidates(*, events, **_kwargs):
        radar_events = [event for event in events if event.get("kind") == "radar"]
        radar_frame_ids_by_call.append(
            [int(event["candidates"]["frame_index"].iloc[0]) for event in radar_events]
        )
        return [event["candidates"].iloc[0].copy() for event in radar_events]

    monkeypatch.setattr(fixed_lag, "_select_tracklet_viterbi_path", select_first_candidates)

    selected = fixed_lag.select_fixed_lag_tracklet_viterbi_path(
        events=[_radar_event(0), _radar_event(1)],
        anchors={},
        covariance=np.eye(3),
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(),
        lag_s=1.0,
    )

    assert radar_frame_ids_by_call == [[0, 1], [0, 1]]
    assert len(selected) == 2
    assert int(selected[1]["association_lag_window_event_count"]) == 2
    assert int(selected[1]["association_lag_window_radar_count"]) == 2
