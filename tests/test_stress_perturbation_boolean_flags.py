from __future__ import annotations

import pandas as pd

from raft_uav.stress.perturbations import PerturbationConfig, perturb_radar


def test_false_track_injection_preserves_csv_false_flags() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0],
            "frame_index": [0],
            "track_id": [1],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
            "cat_prob_uav": [0.9],
            "stress_false_track": ["False"],
        }
    )

    output = perturb_radar(
        radar,
        PerturbationConfig(name="false", false_tracks_per_frame=1, seed=1),
    )

    original_flag = output.loc[output["track_id"] == 1, "stress_false_track"].iloc[0]
    injected_flag = output.loc[output["track_id"] == 2, "stress_false_track"].iloc[0]
    assert not bool(original_flag)
    assert bool(injected_flag)
