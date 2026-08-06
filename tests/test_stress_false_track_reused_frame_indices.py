from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import (
    drop_radar_frames,
    inject_false_tracks,
    jitter_timestamps,
)


class _AlternatingRng:
    def random(self, size: int) -> np.ndarray:
        assert size == 2
        return np.asarray([0.0, 1.0])


class _DistinctJitterRng:
    def normal(self, loc: float, scale: float, size: int) -> np.ndarray:
        assert loc == 0.0
        assert scale == 1.0
        assert size == 4
        return np.asarray([1.0, 9.0, 2.0, 8.0])


def _reused_frame_radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * 4,
            "frame_index": [7] * 4,
            "time_s": [0.0, 0.0, 10.0, 10.0],
            "track_id": [1, 2, 1, 2],
            "east_m": [0.0, 0.0, 100.0, 100.0],
            "north_m": [0.0] * 4,
            "up_m": [0.0] * 4,
            "cat_prob_uav": [0.9, 0.8, 0.7, 0.6],
        }
    )


def test_false_track_injection_separates_reused_frame_indices() -> None:
    perturbed = inject_false_tracks(
        _reused_frame_radar(),
        false_tracks_per_frame=1,
        position_std_m=0.0,
        rng=np.random.default_rng(0),
    )

    synthetic = perturbed.loc[perturbed["stress_false_track"]].sort_values("time_s")
    assert synthetic["time_s"].tolist() == [0.0, 10.0]
    assert synthetic["east_m"].tolist() == [0.0, 100.0]
    assert synthetic["track_id"].tolist() == [3, 4]


def test_radar_dropout_separates_reused_frame_indices() -> None:
    retained = drop_radar_frames(
        _reused_frame_radar(),
        rate=0.5,
        rng=_AlternatingRng(),
    )

    assert retained["time_s"].tolist() == [10.0, 10.0]
    assert retained["track_id"].tolist() == [1, 2]


def test_timestamp_jitter_separates_reused_frame_indices() -> None:
    jittered = jitter_timestamps(
        _reused_frame_radar(),
        std_s=1.0,
        rng=_DistinctJitterRng(),
    )

    assert jittered["time_s"].tolist() == [1.0, 1.0, 12.0, 12.0]
