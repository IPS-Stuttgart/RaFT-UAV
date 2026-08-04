from __future__ import annotations

import pandas as pd

from raft_uav.research import candidate_set_recall as package_candidate_set_recall
from raft_uav.research.diagnostics import (
    candidate_set_recall as module_candidate_set_recall,
)

_RECALL_FUNCTIONS = (package_candidate_set_recall, module_candidate_set_recall)


def _truth(times: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": times,
            "east_m": times,
            "north_m": [0.0] * len(times),
            "up_m": [0.0] * len(times),
        }
    )


def test_candidate_recall_normalizes_serialized_frame_chronology() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [10.0, 1.0, 2.0],
            "frame_index": [10, 1, 2],
            "track_id": [110, 101, 102],
            "east_m": [10.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )
    serialized = radar.assign(
        time_s=radar["time_s"].astype(str),
        frame_index=radar["frame_index"].astype(str),
        track_id=radar["track_id"].astype(str),
    )

    for candidate_set_recall in _RECALL_FUNCTIONS:
        expected = candidate_set_recall(
            radar,
            _truth([1.0, 2.0, 10.0]),
            distance_gate_m=0.01,
            max_time_delta_s=0.0,
        )
        actual = candidate_set_recall(
            serialized,
            _truth([1.0, 2.0, 10.0]),
            distance_gate_m=0.01,
            max_time_delta_s=0.0,
        )

        assert actual["event_key"].tolist() == expected["event_key"].tolist()
        assert actual["event_key"].tolist() == [
            "frame_index:1",
            "frame_index:2",
            "frame_index:10",
        ]
        assert actual["time_s"].tolist() == [1.0, 2.0, 10.0]
        assert actual["target_present"].tolist() == [True, True, True]


def test_candidate_recall_sorts_serialized_track_ids_numerically() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [1.0, 1.0],
            "frame_index": [1, 1],
            "track_id": [10, 2],
            "east_m": [100.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    serialized = radar.assign(
        time_s=radar["time_s"].astype(str),
        frame_index=radar["frame_index"].astype(str),
        track_id=radar["track_id"].astype(str),
    )

    for candidate_set_recall in _RECALL_FUNCTIONS:
        result = candidate_set_recall(
            serialized,
            _truth([1.0]),
            distance_gate_m=0.01,
            max_time_delta_s=0.0,
            preselector=lambda frame: frame.head(1),
        )

        assert result["candidate_count"].tolist() == [1]
        assert result["best_candidate_error_m"].tolist() == [0.0]
        assert result["target_present"].tolist() == [True]
