from __future__ import annotations

import pandas as pd

from raft_uav.stress.perturbations import PerturbationConfig, perturb_radar


def test_false_track_injection_is_scoped_by_sequence() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_id": [1, 2],
            "east_m": [0.0, 1000.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.9, 0.9],
        }
    )

    perturbed = perturb_radar(
        radar,
        PerturbationConfig(
            name="false_tracks",
            false_tracks_per_frame=1,
            false_track_position_std_m=0.0,
            seed=1,
        ),
    )

    false_tracks = perturbed.loc[perturbed["stress_false_track"]].sort_values(
        "sequence_id"
    )
    assert len(false_tracks) == 2
    assert false_tracks["sequence_id"].tolist() == ["seqA", "seqB"]
    assert false_tracks["east_m"].tolist() == [0.0, 1000.0]
