from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.tracker import TrackerConfig, select_tracklet_path


@pytest.mark.parametrize("track_id", ["", " ", "\t"])
def test_tracker_does_not_treat_blank_track_ids_as_stable_tracklets(
    track_id: str,
) -> None:
    candidates = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["radar"] * 4,
            "track_id": [track_id] * 4,
            "x_m": [0.0, 100.0, 1.0, 101.0],
            "y_m": [0.0] * 4,
            "z_m": [0.0] * 4,
            "confidence": [1.0, 0.1, 1.0, 0.1],
        }
    )

    selected = select_tracklet_path(
        candidates,
        config=TrackerConfig(
            selection_mobility_radius_m=0.0,
            selection_motion_weight=1.0,
            selection_confidence_weight=1.0,
        ),
    )

    assert selected["time_s"].tolist() == [0.0, 1.0]
    assert selected["x_m"].tolist() == [0.0, 1.0]
