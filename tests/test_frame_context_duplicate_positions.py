from __future__ import annotations

import pandas as pd

from raft_uav.research.tracklet_models import frame_context_features


def test_frame_context_preserves_zero_distance_between_distinct_candidates() -> None:
    candidates = pd.DataFrame(
        {
            "east_m": [0.0, 0.0, 10.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    context = frame_context_features(candidates)

    assert context["nearest_neighbor_distance_m"].tolist() == [0.0, 0.0, 10.0]
    assert context["mean_neighbor_distance_m"].tolist() == [5.0, 5.0, 10.0]
