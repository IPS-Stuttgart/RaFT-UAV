from __future__ import annotations

import pandas as pd

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_retention import (
    _track_support_by_id,
    _track_support_cost,
)


def test_track_support_ignores_fractional_and_boolean_track_ids() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [12.75, 12.75, True, True, "9", 9.0, "9.0"],
            "frame_index": list(range(7)),
            "time_s": [float(index) for index in range(7)],
            "cat_prob_uav": [0.5] * 7,
        }
    )

    support = _track_support_by_id(radar)

    assert set(support) == {9}
    assert support[9]["count"] == 3.0


def test_fractional_history_does_not_reward_integer_track() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [12.75, 12.75, 12.75],
            "frame_index": [0, 1, 2],
            "time_s": [0.0, 1.0, 2.0],
            "cat_prob_uav": [0.9, 0.9, 0.9],
        }
    )
    support = _track_support_by_id(radar)
    config = TrackletViterbiAssociationConfig(range_gate_m=None)

    cost, diagnostics = _track_support_cost(
        pd.Series({"track_id": 12}),
        track_support_by_id=support,
        config=config,
    )

    assert support == {}
    assert cost == 0.0
    assert diagnostics == {}
