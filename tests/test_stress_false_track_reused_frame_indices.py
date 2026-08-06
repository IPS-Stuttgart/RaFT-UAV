from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import inject_false_tracks


def test_false_track_injection_separates_reused_frame_indices() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "frame_index": [7, 7],
            "time_s": [0.0, 10.0],
            "track_id": [1, 2],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.9, 0.8],
        }
    )

    perturbed = inject_false_tracks(
        radar,
        false_tracks_per_frame=1,
        position_std_m=0.0,
        rng=np.random.default_rng(0),
    )

    synthetic = perturbed.loc[perturbed["stress_false_track"]].sort_values("time_s")
    assert synthetic["time_s"].tolist() == [0.0, 10.0]
    assert synthetic["east_m"].tolist() == [0.0, 100.0]
    assert synthetic["track_id"].tolist() == [3, 4]
