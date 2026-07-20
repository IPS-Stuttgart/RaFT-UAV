from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import inject_false_tracks


def test_false_track_injection_preserves_arbitrary_precision_track_ids() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_id": ["18446744073709551616", "18446744073709551617"],
            "east_m": [0.0, 1.0],
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

    synthetic = perturbed.loc[perturbed["stress_false_track"]]
    assert synthetic["track_id"].tolist() == [18446744073709551618]
