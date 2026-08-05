from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.research.tracklet_models import (
    estimate_frame_clutter_density,
    tracklet_feature_frame,
)


def test_tracklet_features_keep_reused_track_ids_sequence_local() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["a", "a", "b", "b"],
            "track_id": [7, 7, 7, 7],
            "frame_index": [0, 1, 0, 1],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "east_m": [0.0, 1.0, 100.0, 102.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
        }
    )

    features = (
        tracklet_feature_frame(radar)
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )

    assert features["sequence_id"].tolist() == ["a", "b"]
    assert features["frames"].tolist() == [2, 2]
    assert features["mean_speed_mps"].tolist() == pytest.approx([1.0, 2.0])


def test_clutter_density_counts_reused_frame_indices_per_sequence() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["a", "b", "b", "b"],
            "frame_index": [0, 0, 0, 0],
            "time_s": [0.0, 0.0, 0.0, 0.0],
        }
    )

    summary = estimate_frame_clutter_density(radar)

    assert summary["mean_candidates_per_frame"] == pytest.approx(2.0)
    assert summary["p95_candidates_per_frame"] == pytest.approx(2.9)
