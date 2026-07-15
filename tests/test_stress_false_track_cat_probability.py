from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import PerturbationConfig, perturb_radar


def test_false_track_injection_normalizes_malformed_category_probabilities() -> None:
    category_probabilities = [None, "invalid", np.nan, np.inf, -0.5, 0.1, 0.9]
    radar = pd.DataFrame(
        {
            "time_s": np.arange(len(category_probabilities), dtype=float),
            "frame_index": np.arange(len(category_probabilities), dtype=int),
            "track_id": np.ones(len(category_probabilities), dtype=int),
            "east_m": np.zeros(len(category_probabilities)),
            "north_m": np.zeros(len(category_probabilities)),
            "up_m": np.zeros(len(category_probabilities)),
            "cat_prob_uav": category_probabilities,
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

    false_tracks = perturbed.loc[perturbed["stress_false_track"]].sort_values("frame_index")
    actual = false_tracks["cat_prob_uav"].to_numpy(dtype=float)
    np.testing.assert_allclose(actual, [0.2, 0.2, 0.2, 0.2, 0.0, 0.1, 0.2])
    assert np.isfinite(actual).all()
