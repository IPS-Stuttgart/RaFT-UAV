from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import (
    PerturbationConfig,
    drop_radar_frames,
    perturb_radar,
)


def _partially_indexed_radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "frame_index": [10.0, 10.0, np.nan, np.nan],
            "track_id": [1, 2, 1, 2],
            "east_m": [0.0, 1.0, 2.0, 3.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
            "cat_prob_uav": [0.9, 0.8, 0.7, 0.6],
        }
    )


def test_radar_drop_uses_time_when_frame_index_is_incomplete() -> None:
    dropped = drop_radar_frames(
        _partially_indexed_radar(),
        rate=1.0,
        rng=np.random.default_rng(2),
    )

    assert dropped.empty


def test_false_track_injection_covers_partially_indexed_frames() -> None:
    perturbed = perturb_radar(
        _partially_indexed_radar(),
        PerturbationConfig(
            name="partial-frame-index",
            false_tracks_per_frame=1,
            false_track_position_std_m=0.0,
            seed=3,
        ),
    )

    false_rows = perturbed.loc[perturbed["stress_false_track"]]
    assert len(perturbed) == 6
    assert false_rows.groupby(["sequence_id", "time_s"]).size().to_dict() == {
        ("seqA", 0.0): 1,
        ("seqA", 1.0): 1,
    }
    assert false_rows.sort_values("time_s")["east_m"].tolist() == [0.5, 2.5]
