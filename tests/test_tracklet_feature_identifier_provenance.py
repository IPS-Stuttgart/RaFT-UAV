from __future__ import annotations

import pandas as pd

from raft_uav.research.tracklet_models import tracklet_feature_frame


def _two_track_radar(track_ids: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": track_ids,
            "frame_index": [0, 1, 0, 1],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "east_m": [0.0, 1.0, 10.0, 11.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_tracklet_features_preserve_opaque_and_zero_padded_ids() -> None:
    radar = _two_track_radar(["001", "001", "uav-A", "uav-A"])

    features = tracklet_feature_frame(radar)

    assert features["track_id"].tolist() == ["001", "uav-A"]
    assert features["frames"].tolist() == [2, 2]


def test_tracklet_features_preserve_fractional_ids_without_collision() -> None:
    radar = _two_track_radar([1.25, 1.25, 1.75, 1.75])

    features = tracklet_feature_frame(radar)

    assert features["track_id"].tolist() == [1.25, 1.75]
    assert features["frames"].tolist() == [2, 2]


def test_tracklet_features_preserve_ids_through_flight_scope() -> None:
    radar = _two_track_radar(["001", "001", "uav-A", "uav-A"])
    radar["flight_id"] = ["flight-a", "flight-a", "flight-b", "flight-b"]

    features = tracklet_feature_frame(radar)

    assert features[["flight_id", "track_id"]].values.tolist() == [
        ["flight-a", "001"],
        ["flight-b", "uav-A"],
    ]
    assert features["frames"].tolist() == [2, 2]
