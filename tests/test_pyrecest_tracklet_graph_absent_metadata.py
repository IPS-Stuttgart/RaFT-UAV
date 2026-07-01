from __future__ import annotations

import numpy as np

from raft_uav.baselines.pyrecest_tracklet_graph import (
    FortemTrackletGraphConfig,
    Tracklet,
    rank_fortem_tracklet_paths,
)


def test_rank_fortem_tracklet_paths_accepts_tracklet_without_aux_fields() -> None:
    tracklet = Tracklet(
        id="solo",
        start_time=0.0,
        end_time=1.0,
        start_state=np.array([0.0, 0.0, 0.0], dtype=float),
        end_state=np.array([1.0, 0.0, 0.0], dtype=float),
        cost=2.0,
        metadata={},
    )

    paths = rank_fortem_tracklet_paths(
        [tracklet],
        config=FortemTrackletGraphConfig(top_k_paths=1, coverage_reward_per_row=1.0),
    )

    assert paths[0].tracklet_ids == ("solo",)
    assert paths[0].cost == 1.0
