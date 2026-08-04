from __future__ import annotations

import pandas as pd

from raft_uav.baselines.learned_radar_likelihood import (
    estimate_stateful_transition_costs,
)


def test_stateful_transition_costs_sort_equal_time_frames_numerically() -> None:
    examples = pd.DataFrame(
        {
            "flight": ["track"] * 3 + ["miss"] * 3,
            "frame_index": [1, 2, 10, 1, 2, 10],
            "time_s": [5.0] * 6,
            "label": [1, 1, 1, 0, 1, 0],
            "track_id": [7, 7, 8, 20, 20, 20],
        }
    )

    costs = estimate_stateful_transition_costs(examples)

    assert costs["recoveries_after_miss"] == 1
    assert costs["continued_misses"] == 0
    assert costs["same_positive_track_transitions"] == 1
    assert costs["switched_positive_track_transitions"] == 1
