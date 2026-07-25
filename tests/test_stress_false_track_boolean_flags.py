from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.stress.perturbations import (
    PerturbationConfig,
    inject_false_tracks,
    perturb_radar,
)


def _radar_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * 4,
            "time_s": [0.0] * 4,
            "frame_index": [0] * 4,
            "track_id": [10, 11, 12, 13],
            "east_m": [0.0, 1.0, 2.0, 3.0],
            "north_m": [0.0] * 4,
            "up_m": [0.0] * 4,
            "cat_prob_uav": [0.9] * 4,
            "stress_false_track": ["False", "0", " TRUE ", None],
        }
    )


def test_perturb_radar_preserves_serialized_false_track_flag_semantics() -> None:
    perturbed = perturb_radar(
        _radar_rows(),
        PerturbationConfig(
            name="serialized_flags",
            false_tracks_per_frame=1,
            false_track_position_std_m=0.0,
            seed=7,
        ),
    ).set_index("track_id")

    assert perturbed["stress_false_track"].dtype == bool
    assert bool(perturbed.loc[10, "stress_false_track"]) is False
    assert bool(perturbed.loc[11, "stress_false_track"]) is False
    assert bool(perturbed.loc[12, "stress_false_track"]) is True
    assert bool(perturbed.loc[13, "stress_false_track"]) is False
    assert bool(perturbed.loc[14, "stress_false_track"]) is True


def test_false_track_injection_rejects_ambiguous_persisted_flags() -> None:
    radar = _radar_rows().iloc[:1].copy()
    radar.index = [42]
    radar.loc[42, "stress_false_track"] = "maybe"

    with pytest.raises(
        ValueError,
        match=r"stress_false_track contains invalid Boolean values at rows \[42\]",
    ):
        inject_false_tracks(
            radar,
            false_tracks_per_frame=1,
            position_std_m=0.0,
            rng=np.random.default_rng(3),
        )
