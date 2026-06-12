import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns


def test_candidate_normalizer_accepts_case_insensitive_canonical_columns():
    raw = pd.DataFrame(
        {
            "Sequence_ID": ["seqA"],
            "Time_S": [1.25],
            "Source": ["radar"],
            "X_M": [10.0],
            "Y_M": [20.0],
            "Z_M": [30.0],
        }
    )

    rows = normalize_candidate_columns(raw)

    assert rows.loc[0, "sequence_id"] == "seqA"
    assert rows.loc[0, "time_s"] == 1.25
    assert rows.loc[0, "source"] == "radar"
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [10.0, 20.0, 30.0]


def test_truth_normalizer_accepts_case_insensitive_canonical_columns():
    raw = pd.DataFrame(
        {
            "Sequence_ID": ["seqB"],
            "Time_S": [2.5],
            "X_M": [1.0],
            "Y_M": [2.0],
            "Z_M": [3.0],
        }
    )

    rows = normalize_truth_columns(raw)

    assert rows.loc[0, "sequence_id"] == "seqB"
    assert rows.loc[0, "time_s"] == 2.5
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


def test_candidate_normalizer_accepts_flattened_ros_pose_columns():
    raw = pd.DataFrame(
        {
            "sequence": ["seqC"],
            "header.stamp.sec": [3],
            "header.stamp.nanosec": [250_000_000],
            "sensor": ["detector"],
            "child_frame_id": ["uav_1"],
            "pose.pose.position.x": [4.0],
            "pose.pose.position.y": [5.0],
            "pose.pose.position.z": [6.0],
        }
    )

    rows = normalize_candidate_columns(raw)

    assert rows.loc[0, "sequence_id"] == "seqC"
    assert abs(float(rows.loc[0, "time_s"]) - 3.25) < 1.0e-12
    assert rows.loc[0, "source"] == "detector"
    assert rows.loc[0, "track_id"] == "uav_1"
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [4.0, 5.0, 6.0]


def test_truth_normalizer_accepts_flattened_detection3d_bbox_columns():
    raw = pd.DataFrame(
        {
            "sequence_id": ["seqD"],
            "stamp.sec": [4],
            "stamp.nsec": [500_000_000],
            "bbox.center.position.x": [7.0],
            "bbox.center.position.y": [8.0],
            "bbox.center.position.z": [9.0],
        }
    )

    rows = normalize_truth_columns(raw)

    assert rows.loc[0, "sequence_id"] == "seqD"
    assert abs(float(rows.loc[0, "time_s"]) - 4.5) < 1.0e-12
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [7.0, 8.0, 9.0]
