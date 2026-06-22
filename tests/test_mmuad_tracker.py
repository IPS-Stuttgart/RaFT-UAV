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


def test_viterbi_selection_rejects_high_confidence_temporal_outlier():
    candidates = pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "start",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 5.0,
                "confidence": 1.0,
            },
            {
                "time_s": 1.0,
                "source": "lidar_360",
                "track_id": "true",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 5.0,
                "confidence": 0.2,
            },
            {
                "time_s": 1.0,
                "source": "lidar_360",
                "track_id": "outlier",
                "x_m": 50.0,
                "y_m": 0.0,
                "z_m": 5.0,
                "confidence": 1.0,
            },
            {
                "time_s": 2.0,
                "source": "lidar_360",
                "track_id": "end",
                "x_m": 2.0,
                "y_m": 0.0,
                "z_m": 5.0,
                "confidence": 1.0,
            },
        ]
    )

    selected = select_tracklet_path(
        candidates,
        config=TrackerConfig(
            selection_mode="viterbi",
            viterbi_motion_weight=8.0,
            viterbi_ranker_weight=1.0,
            viterbi_max_speed_mps=20.0,
        ),
    )

    assert selected.sort_values("time_s")["track_id"].tolist() == ["start", "true", "end"]
    assert selected["selected_path_mode"].unique().tolist() == ["viterbi"]


def test_viterbi_selection_source_switch_penalty_prefers_continuing_source():
    candidates = pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "lidar0",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 5.0,
                "confidence": 1.0,
            },
            {
                "time_s": 1.0,
                "source": "radar_enhance_pcl",
                "track_id": "radar1",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 5.0,
                "confidence": 1.0,
            },
            {
                "time_s": 1.0,
                "source": "lidar_360",
                "track_id": "lidar1",
                "x_m": 1.1,
                "y_m": 0.0,
                "z_m": 5.0,
                "confidence": 1.0,
            },
        ]
    )

    selected = select_tracklet_path(
        candidates,
        config=TrackerConfig(
            selection_mode="viterbi",
            viterbi_source_switch_penalty=1.0,
        ),
    )

    assert selected.sort_values("time_s")["source"].tolist() == ["lidar_360", "lidar_360"]
