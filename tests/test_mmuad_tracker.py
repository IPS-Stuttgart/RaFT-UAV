import pandas as pd

from raft_uav.mmuad.tracker import TrackerConfig, select_tracklet_path


def test_select_tracklet_path_uses_longest_source_priority_prefix():
    candidates = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "source": ["lidar", "lidar-cluster"],
            "track_id": ["plain", "cluster"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [0.5, 1.0],
        }
    )
    config = TrackerConfig(source_priority=("lidar", "lidar-cluster"))

    selected = select_tracklet_path(candidates, config=config)

    assert selected.loc[0, "source"] == "lidar"
    assert selected.loc[0, "track_id"] == "plain"
