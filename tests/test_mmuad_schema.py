import pandas as pd
import pytest

from raft_uav.mmuad.schema import (
    normalize_candidate_columns,
    normalize_time_column_aliases,
    normalize_truth_columns,
)


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


@pytest.mark.parametrize(
    ("seconds_col", "nanoseconds_col"),
    (
        ("stamp.secs", "stamp.nsecs"),
        ("header.stamp.secs", "header.stamp.nsecs"),
        ("timestamp.secs", "timestamp.nsecs"),
    ),
)
def test_time_normalizer_accepts_ros1_plural_flattened_stamp_pairs(
    seconds_col: str,
    nanoseconds_col: str,
):
    raw = pd.DataFrame({seconds_col: [7], nanoseconds_col: [125_000_000]})

    rows = normalize_time_column_aliases(raw)

    assert abs(float(rows.loc[0, "time_s"]) - 7.125) < 1.0e-12


def test_truth_normalizer_accepts_ros1_plural_flattened_header_stamp_columns():
    raw = pd.DataFrame(
        {
            "sequence_id": ["seqRos1"],
            "header.stamp.secs": [7],
            "header.stamp.nsecs": [125_000_000],
            "pose.position.x": [1.0],
            "pose.position.y": [2.0],
            "pose.position.z": [3.0],
        }
    )

    rows = normalize_truth_columns(raw)

    assert rows.loc[0, "sequence_id"] == "seqRos1"
    assert abs(float(rows.loc[0, "time_s"]) - 7.125) < 1.0e-12
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


def test_candidate_normalizer_uses_flattened_ros_frame_as_source():
    raw = pd.DataFrame(
        {
            "sequence_id": ["seqFrame"],
            "time_s": [1.0],
            "header.frame_id": ["detector_frame"],
            "child_frame_id": ["uav_2"],
            "position.x": [1.0],
            "position.y": [2.0],
            "position.z": [3.0],
        }
    )

    rows = normalize_candidate_columns(raw)

    assert rows.loc[0, "source"] == "detector_frame"
    assert rows.loc[0, "track_id"] == "uav_2"
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


def test_candidate_normalizer_accepts_flattened_detection_result_columns():
    raw = pd.DataFrame(
        {
            "sequence_id": ["seqResult"],
            "time_s": [2.0],
            "source": ["detector"],
            "bbox.center.position.x": [1.0],
            "bbox.center.position.y": [2.0],
            "bbox.center.position.z": [3.0],
            "results.0.hypothesis.class_id": ["Mavic3"],
            "results.0.hypothesis.score": [0.82],
        }
    )

    rows = normalize_candidate_columns(raw)

    assert rows.loc[0, "class_name"] == "Mavic3"
    assert abs(float(rows.loc[0, "confidence"]) - 0.82) < 1.0e-12
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


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
