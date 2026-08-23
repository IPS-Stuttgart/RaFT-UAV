from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import (
    drop_radar_frames,
    drop_rf_bursts,
    inject_false_tracks,
    jitter_timestamps,
)


def test_false_track_injection_is_scoped_by_flight_id() -> None:
    radar = pd.DataFrame(
        {
            "flight_id": ["f1", "f2"],
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_id": [1, 2],
            "east_m": [0.0, 1000.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.9, 0.9],
        }
    )

    perturbed = inject_false_tracks(
        radar,
        false_tracks_per_frame=1,
        position_std_m=0.0,
        rng=np.random.default_rng(1),
    )

    false_tracks = perturbed.loc[perturbed["stress_false_track"]].sort_values(
        "flight_id"
    )
    assert len(false_tracks) == 2
    assert false_tracks["flight_id"].tolist() == ["f1", "f2"]
    assert false_tracks["east_m"].tolist() == [0.0, 1000.0]


def test_false_track_injection_uses_joint_sequence_and_flight_scope() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "flight_id": ["f1", "f2"],
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_id": [1, 2],
            "east_m": [0.0, 1000.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    perturbed = inject_false_tracks(
        radar,
        false_tracks_per_frame=1,
        position_std_m=0.0,
        rng=np.random.default_rng(1),
    )

    false_tracks = perturbed.loc[perturbed["stress_false_track"]].sort_values(
        "flight_id"
    )
    assert len(false_tracks) == 2
    assert false_tracks["east_m"].tolist() == [0.0, 1000.0]


def test_radar_frame_drop_uses_joint_sequence_and_flight_scope() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "flight_id": ["f1", "f2"],
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
        }
    )

    perturbed = drop_radar_frames(
        radar,
        rate=0.5,
        rng=np.random.default_rng(0),
    )

    assert perturbed["flight_id"].tolist() == ["f1"]


def test_radar_frame_drop_uses_time_when_frame_index_is_missing() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "flight_id": ["f1", "f1"],
            "time_s": [1.0, 2.0],
            "frame_index": [np.nan, np.nan],
        }
    )

    perturbed = drop_radar_frames(
        radar,
        rate=1.0,
        rng=np.random.default_rng(0),
    )

    assert perturbed.empty


def test_rf_burst_drop_is_scoped_by_flight_id() -> None:
    rf = pd.DataFrame(
        {
            "flight_id": ["f1", "f2"],
            "time_s": [0.0, 0.0],
        }
    )

    perturbed = drop_rf_bursts(
        rf,
        rate=0.5,
        rng=np.random.default_rng(0),
    )

    assert perturbed["flight_id"].tolist() == ["f1"]


def test_timestamp_jitter_is_scoped_by_flight_id() -> None:
    radar = pd.DataFrame(
        {
            "flight_id": ["f1", "f2"],
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
        }
    )

    perturbed = jitter_timestamps(
        radar,
        std_s=1.0,
        rng=np.random.default_rng(0),
    )

    assert perturbed.loc[0, "time_s"] != perturbed.loc[1, "time_s"]
