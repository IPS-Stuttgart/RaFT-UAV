from __future__ import annotations

import gzip
import io
import json
import struct
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from raft_uav.coordinates import LocalENUProjector
from raft_uav.mmuad.archive import extract_mmuad_archive
from raft_uav.mmuad.calibration import (
    load_calibration_file,
    load_calibration_json,
    transform_candidate_frame,
)
from raft_uav.mmuad.cli import main as mmuad_cli_main
from raft_uav.mmuad.completion import complete_results_to_truth_timestamps
from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
    load_mmaud_results_csv,
    load_mmaud_results_file,
    write_evaluation_artifacts,
)
from raft_uav.mmuad.evaluator import validate_mmaud_results_frame
from raft_uav.mmuad.evaluate import evaluate_submission_csv
from raft_uav.mmuad.inspect import inspect_sequence_root, write_layout_report
from raft_uav.mmuad.io import (
    load_candidate_file,
    load_candidate_csv,
    load_point_cloud_csv_as_candidates,
    load_point_cloud_file_as_candidates,
    load_truth_file,
    load_truth_csv,
    merge_candidate_frames,
)
from raft_uav.mmuad.layout import inspect_mmuad_layout
from raft_uav.mmuad.mot import (
    MultiObjectTrackerConfig,
    compute_multi_object_metrics,
    run_mmuad_multi_object_tracker,
)
from raft_uav.mmuad.native_ros import (
    bounding_box3d_message_to_rows,
    detection2d_message_to_rows,
    detection3d_message_to_rows,
    extract_native_rosbag_topic_map,
    geodetic_message_to_rows,
    livox_custom_message_to_points,
    marker_message_to_rows,
    multidof_message_to_rows,
    position_message_to_row,
    position_message_to_rows,
    radar_polar_message_to_rows,
    tracked_objects_message_to_rows,
)
from raft_uav.mmuad.pointcloud2 import pointcloud2_to_candidates, pointcloud2_to_dataframe
from raft_uav.mmuad.rosbag_bridge import (
    inspect_rosbag,
    load_topic_map_exports,
    write_topic_map_template,
)
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame
from raft_uav.mmuad.sequence import (
    discover_sequence_paths,
    load_sequence_export,
    official_track5_timestamp_template,
)
from raft_uav.mmuad.splits import filter_sequences_by_split, load_split_manifest
from raft_uav.mmuad.submission import (
    compute_trajectory_metrics,
    estimates_to_mmaud_results_frame,
    estimates_to_submission_frame,
    inspect_submission_zip,
    load_sequence_class_map,
    validate_official_track5_submission,
    write_mmaud_results_csv,
    write_submission_json,
    write_submission_zip,
    write_ug2_codabench_zip,
)
from raft_uav.mmuad.tracker import (
    TrackerConfig,
    add_truth_errors,
    compute_metrics,
    run_mmuad_tracker,
    write_tracker_output,
)


def test_candidate_loader_accepts_aliases(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "seq": ["s1"],
            "timestamp_s": [0.0],
            "sensor": ["radar"],
            "id": [7],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "score": [0.9],
        }
    ).to_csv(path, index=False)
    frame = load_candidate_csv(path)
    assert frame.rows.loc[0, "sequence_id"] == "s1"
    assert frame.rows.loc[0, "source"] == "radar"
    assert frame.rows.loc[0, "track_id"] == 7


def test_candidate_loader_accepts_nanosecond_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "timestamp_ns": [1_500_000_000],
            "source": ["radar"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    ).to_csv(path, index=False)

    frame = load_candidate_csv(path)

    assert abs(float(frame.rows.loc[0, "time_s"]) - 1.5) < 1e-12


def test_candidate_csv_loader_uses_default_source_hint(tmp_path: Path) -> None:
    path = tmp_path / "detections.csv"
    pd.DataFrame(
        {
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    ).to_csv(path, index=False)

    frame = load_candidate_file(path, source="camera_front")

    assert frame.rows.loc[0, "source"] == "camera_front"


def test_candidate_csv_loader_infers_flattened_ros_frame_source(tmp_path: Path) -> None:
    path = tmp_path / "ros_pose_candidates.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "header.stamp.sec": [3],
            "header.stamp.nanosec": [250_000_000],
            "header.frame_id": ["detector_frame"],
            "child_frame_id": ["uav_1"],
            "pose.pose.position.x": [1.0],
            "pose.pose.position.y": [2.0],
            "pose.pose.position.z": [3.0],
        }
    ).to_csv(path, index=False)

    frame = load_candidate_file(path)

    row = frame.rows.iloc[0]
    assert abs(float(row["time_s"]) - 3.25) < 1e-12
    assert row["source"] == "detector_frame"
    assert row["track_id"] == "uav_1"
    assert row[["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


def test_candidate_csv_loader_explicit_source_overrides_ros_frame_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ros_pose_candidates.csv"
    pd.DataFrame(
        {
            "time_s": [0.0],
            "frame_id": ["world"],
            "child_frame_id": ["uav_2"],
            "position.x": [4.0],
            "position.y": [5.0],
            "position.z": [6.0],
        }
    ).to_csv(path, index=False)

    frame = load_candidate_file(path, source="radar0")

    row = frame.rows.iloc[0]
    assert row["source"] == "radar0"
    assert row["track_id"] == "uav_2"
    assert row[["x_m", "y_m", "z_m"]].tolist() == [4.0, 5.0, 6.0]


def test_candidate_json_loader_accepts_nested_rows(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "sequence_id": "s1",
                        "timestamp_ms": 1500,
                        "sensor": "radar",
                        "id": "track7",
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    frame = load_candidate_file(path)

    assert frame.rows.loc[0, "sequence_id"] == "s1"
    assert frame.rows.loc[0, "source"] == "radar"
    assert frame.rows.loc[0, "track_id"] == "track7"
    assert abs(float(frame.rows.loc[0, "time_s"]) - 1.5) < 1e-12


def test_candidate_json_loader_flattens_ros_pose_rows(tmp_path: Path) -> None:
    path = tmp_path / "pose_candidates.json"
    path.write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "header": {
                            "stamp": {"sec": 2, "nanosec": 250_000_000},
                            "frame_id": "radar_frame",
                        },
                        "child_frame_id": "uav_7",
                        "pose": {
                            "pose": {
                                "position": {"x": 1.0, "y": 2.0, "z": 3.0}
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    frame = load_candidate_file(path)

    row = frame.rows.iloc[0]
    assert abs(float(row["time_s"]) - 2.25) < 1e-12
    assert row["source"] == "radar_frame"
    assert row["track_id"] == "uav_7"
    assert row[["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]


def test_candidate_json_loader_flattens_detection3d_bbox_rows(tmp_path: Path) -> None:
    path = tmp_path / "detection3d_candidates.json"
    path.write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "header": {
                            "stamp": {"sec": 4, "nanosec": 500_000_000},
                            "frame_id": "detector_frame",
                        },
                        "id": "det-1",
                        "bbox": {
                            "center": {
                                "position": {"x": 4.0, "y": 5.0, "z": 6.0}
                            }
                        },
                        "results": [
                            {"hypothesis": {"class_id": "uav", "score": 0.8}},
                            {"hypothesis": {"class_id": "bird", "score": 0.2}},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    frame = load_candidate_file(path)

    row = frame.rows.iloc[0]
    assert abs(float(row["time_s"]) - 4.5) < 1e-12
    assert row["source"] == "detector_frame"
    assert row["track_id"] == "det-1"
    assert row[["x_m", "y_m", "z_m"]].tolist() == [4.0, 5.0, 6.0]
    assert float(row["confidence"]) == 0.8
    assert row["class_name"] == "uav"


def test_candidate_jsonl_loader_accepts_row_exports(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    rows = [
        {
            "sequence_id": "s1",
            "timestamp_s": 0.0,
            "sensor": "radar",
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
        },
        {
            "sequence_id": "s1",
            "timestamp_s": 1.0,
            "sensor": "camera",
            "x": 2.0,
            "y": 3.0,
            "z": 4.0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    frame = load_candidate_file(path)

    assert frame.rows["sequence_id"].tolist() == ["s1", "s1"]
    assert frame.rows["source"].tolist() == ["radar", "camera"]
    assert frame.rows["time_s"].tolist() == [0.0, 1.0]


def test_candidate_loader_accepts_gzipped_csv_exports(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv.gz"
    pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "timestamp_s": [2.0],
            "sensor": ["radar"],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        }
    ).to_csv(path, index=False, compression="gzip")

    frame = load_candidate_file(path)

    assert frame.rows.loc[0, "sequence_id"] == "s1"
    assert frame.rows.loc[0, "source"] == "radar"
    assert abs(float(frame.rows.loc[0, "time_s"]) - 2.0) < 1.0e-12


def test_candidate_json_loader_accepts_column_map(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "time_s": [0.0, 1.0],
                "source": "radar",
                "x_m": [1.0, 2.0],
                "y_m": [3.0, 4.0],
                "z_m": [5.0, 6.0],
            }
        ),
        encoding="utf-8",
    )

    frame = load_candidate_file(path)

    assert frame.rows["time_s"].tolist() == [0.0, 1.0]
    assert frame.rows["source"].tolist() == ["radar", "radar"]


def test_truth_loader_accepts_sec_nanosec_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "sec": [2],
            "nanosec": [250_000_000],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    ).to_csv(path, index=False)

    frame = load_truth_csv(path)

    assert abs(float(frame.rows.loc[0, "time_s"]) - 2.25) < 1e-12


def test_truth_json_loader_accepts_sequence_mapping(tmp_path: Path) -> None:
    path = tmp_path / "truth.json"
    path.write_text(
        json.dumps(
            {
                "sequences": {
                    "seq_json": [
                        {
                            "sec": 2,
                            "nanosec": 250_000_000,
                            "x": 1.0,
                            "y": 2.0,
                            "z": 3.0,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    frame = load_truth_file(path)

    assert frame.rows.loc[0, "sequence_id"] == "seq_json"
    assert abs(float(frame.rows.loc[0, "time_s"]) - 2.25) < 1e-12


def test_truth_json_loader_flattens_parent_header_pose_rows(tmp_path: Path) -> None:
    path = tmp_path / "truth_path.json"
    path.write_text(
        json.dumps(
            {
                "header": {
                    "stamp": {"sec": 3, "nanosec": 500_000_000},
                    "frame_id": "world",
                },
                "poses": [
                    {
                        "pose": {
                            "position": {"x": 4.0, "y": 5.0, "z": 6.0}
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    frame = load_truth_file(path, default_sequence_id="seq_path")

    row = frame.rows.iloc[0]
    assert row["sequence_id"] == "seq_path"
    assert abs(float(row["time_s"]) - 3.5) < 1e-12
    assert row[["x_m", "y_m", "z_m"]].tolist() == [4.0, 5.0, 6.0]


def test_truth_ndjson_loader_accepts_row_exports(tmp_path: Path) -> None:
    path = tmp_path / "truth.ndjson"
    rows = [
        {"sequence_id": "s1", "timestamp_s": 0.0, "x": 1.0, "y": 2.0, "z": 3.0},
        {"sequence_id": "s1", "timestamp_s": 1.0, "x": 2.0, "y": 3.0, "z": 4.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    frame = load_truth_file(path)

    assert frame.rows["sequence_id"].tolist() == ["s1", "s1"]
    assert frame.rows["time_s"].tolist() == [0.0, 1.0]


def test_truth_loader_accepts_gzipped_jsonl_exports(tmp_path: Path) -> None:
    path = tmp_path / "truth.jsonl.gz"
    rows = [
        {"sequence_id": "s1", "timestamp_s": 0.0, "x": 1.0, "y": 2.0, "z": 3.0},
        {"sequence_id": "s1", "timestamp_s": 1.0, "x": 2.0, "y": 3.0, "z": 4.0},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row) for row in rows))

    frame = load_truth_file(path)

    assert frame.rows["sequence_id"].tolist() == ["s1", "s1"]
    assert frame.rows["time_s"].tolist() == [0.0, 1.0]


def test_point_cloud_csv_clusters_points(tmp_path: Path) -> None:
    path = tmp_path / "points.csv"
    pd.DataFrame(
        {
            "sequence_id": ["s1"] * 6,
            "time_s": [0.0] * 6,
            "x_m": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "y_m": [0.0, 0.0, 0.1, 10.0, 10.0, 10.1],
            "z_m": [1.0, 1.1, 1.0, 2.0, 2.1, 2.0],
        }
    ).to_csv(path, index=False)
    frame = load_point_cloud_csv_as_candidates(path, voxel_size_m=0.5, min_points=3)
    assert len(frame.rows) == 2
    assert set(frame.rows["source"]) == {"lidar-cluster"}


def test_point_cloud_csv_without_clusters_returns_empty_candidates(tmp_path: Path) -> None:
    path = tmp_path / "points.csv"
    pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 10.0],
            "z_m": [1.0, 2.0],
        }
    ).to_csv(path, index=False)

    frame = load_point_cloud_csv_as_candidates(path, voxel_size_m=0.5, min_points=3)

    assert frame.rows.empty
    frame.validate()


def test_merge_empty_candidate_frames_returns_valid_empty_frame() -> None:
    frame = merge_candidate_frames([CandidateFrame(pd.DataFrame())])

    assert frame.rows.empty
    frame.validate()


def test_tracker_runs_and_writes_metrics(tmp_path: Path) -> None:
    cand_path = tmp_path / "candidates.csv"
    truth_path = tmp_path / "truth.csv"
    rows = []
    truth_rows = []
    for i in range(6):
        t = float(i)
        truth_rows.append({"sequence_id": "s1", "time_s": t, "x_m": t, "y_m": 0.0, "z_m": 2.0})
        rows.append(
            {
                "sequence_id": "s1",
                "time_s": t,
                "source": "radar",
                "track_id": "good",
                "x_m": t,
                "y_m": 0.0,
                "z_m": 2.0,
                "confidence": 1.0,
            }
        )
        rows.append(
            {
                "sequence_id": "s1",
                "time_s": t,
                "source": "lidar",
                "track_id": "bad",
                "x_m": 100.0,
                "y_m": 100.0,
                "z_m": 2.0,
                "confidence": 0.1,
            }
        )
    pd.DataFrame(rows).to_csv(cand_path, index=False)
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    output = run_mmuad_tracker(
        load_candidate_csv(cand_path),
        load_truth_csv(truth_path),
        config=TrackerConfig(soft_anchor_cap_m=1.0),
    )
    assert output.metrics["pooled"]["mean_3d_m"] < 5.0
    paths = write_tracker_output(output, tmp_path / "out")
    assert Path(paths["estimates_csv"]).exists()
    assert json.loads(Path(paths["metrics_json"]).read_text())["pooled"]["mean_3d_m"] < 5.0

def test_tracker_ignores_invalid_manual_candidate_rows() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1", "s1", "s1"],
                "time_s": [0.0, 1.0, 1.0, 2.0],
                "source": ["radar", "bad", "radar", "radar"],
                "track_id": [None, None, None, None],
                "x_m": [0.0, np.nan, 1.0, 2.0],
                "y_m": [0.0, 0.0, 0.0, 0.0],
                "z_m": [2.0, 2.0, 2.0, 2.0],
                "confidence": [1.0, 1.0, 0.1, 1.0],
                "std_xy_m": [10.0, np.nan, 10.0, 10.0],
                "std_z_m": [10.0, np.nan, 10.0, 10.0],
            }
        )
    )
    truth = TruthFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1", "s1"],
                "time_s": [0.0, 1.0, 2.0],
                "x_m": [0.0, 1.0, 2.0],
                "y_m": [0.0, 0.0, 0.0],
                "z_m": [2.0, 2.0, 2.0],
            }
        )
    )

    output = run_mmuad_tracker(candidates, truth, config=TrackerConfig())

    assert output.selected_tracklets["source"].tolist() == ["radar", "radar", "radar"]
    assert output.estimates["source"].tolist() == ["radar", "radar", "radar"]
    assert np.isfinite(
        output.estimates[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(dtype=float)
    ).all()


def test_tracker_accepts_minimal_valid_candidate_frame_without_optional_columns() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1", "s1"],
                "time_s": [0.0, 1.0, 2.0],
                "source": ["radar", "radar", "radar"],
                "x_m": [0.0, 1.0, 2.0],
                "y_m": [0.0, 0.0, 0.0],
                "z_m": [2.0, 2.0, 2.0],
            }
        )
    )
    truth = TruthFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1", "s1"],
                "time_s": [0.0, 1.0, 2.0],
                "x_m": [0.0, 1.0, 2.0],
                "y_m": [0.0, 0.0, 0.0],
                "z_m": [2.0, 2.0, 2.0],
            }
        )
    )

    output = run_mmuad_tracker(candidates, truth, config=TrackerConfig())

    assert output.selected_tracklets["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert output.estimates["update_action"].tolist() == [
        "selected_update",
        "selected_update",
        "selected_update",
    ]
    assert "track_id" in output.selected_tracklets.columns


def test_tracker_replays_only_selected_duplicate_rows_without_track_ids() -> None:
    rows = []
    truth_rows = []
    for time_s in range(3):
        truth_rows.append(
            {
                "sequence_id": "s1",
                "time_s": float(time_s),
                "x_m": float(time_s),
                "y_m": 0.0,
                "z_m": 2.0,
            }
        )
        rows.append(
            {
                "sequence_id": "s1",
                "time_s": float(time_s),
                "source": "candidate",
                "track_id": np.nan,
                "x_m": float(time_s),
                "y_m": 0.0,
                "z_m": 2.0,
                "confidence": 1.0,
                "std_xy_m": 1.0,
                "std_z_m": 1.0,
            }
        )
        rows.append(
            {
                "sequence_id": "s1",
                "time_s": float(time_s),
                "source": "candidate",
                "track_id": np.nan,
                "x_m": 100.0 + float(time_s),
                "y_m": 0.0,
                "z_m": 2.0,
                "confidence": 0.5,
                "std_xy_m": 1.0,
                "std_z_m": 1.0,
            }
        )
    candidates = CandidateFrame(pd.DataFrame(rows))
    truth = TruthFrame(pd.DataFrame(truth_rows))

    output = run_mmuad_tracker(candidates, truth, config=TrackerConfig())

    assert output.estimates["update_action"].tolist().count("selected_update") == 3
    assert output.estimates["update_action"].tolist().count("soft_anchor") == 3
    assert output.selected_tracklets["x_m"].tolist() == [0.0, 1.0, 2.0]
    assert "_candidate_row_id" not in output.selected_tracklets.columns


def test_add_truth_errors_sorts_truth_times() -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [2.0, 2.0, 2.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [2.0, 0.0, 1.0],
            "x_m": [2.0, 0.0, 1.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [2.0, 2.0, 2.0],
        }
    )

    scored = add_truth_errors(estimates, truth)

    assert scored["error_3d_m"].tolist() == [0.0, 0.0, 0.0]


def test_compute_metrics_accepts_3d_errors_without_2d_column() -> None:
    estimates = pd.DataFrame({"error_3d_m": [3.0, "bad", np.nan]})

    metrics = compute_metrics(estimates, truth=None)

    assert metrics["count"] == 1
    assert metrics["mean_3d_m"] == 3.0
    assert metrics["mean_2d_m"] is None


def test_calibration_json_transforms_candidate_coordinates(tmp_path: Path) -> None:
    cand_path = tmp_path / "candidates.csv"
    calib_path = tmp_path / "calibration.json"
    pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "time_s": [1.0],
            "source": ["radar"],
            "track_id": ["a"],
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    ).to_csv(cand_path, index=False)
    calib_path.write_text(
        json.dumps(
            {
                "world_frame": "test_world",
                "sensors": {
                    "radar": {
                        "translation_m": [10.0, 0.0, 1.0],
                        "rpy_deg": [0.0, 0.0, 90.0],
                        "time_offset_s": 0.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    transformed = transform_candidate_frame(
        load_candidate_csv(cand_path),
        load_calibration_json(calib_path),
    )
    row = transformed.rows.iloc[0]
    assert abs(row["x_m"] - 10.0) < 1e-9
    assert abs(row["y_m"] - 1.0) < 1e-9
    assert abs(row["z_m"] - 1.0) < 1e-9
    assert abs(row["time_s"] - 1.5) < 1e-9


def test_sequence_root_discovery_and_loading(tmp_path: Path) -> None:
    seq = tmp_path / "seq001"
    seq.mkdir()
    pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["r1"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    ).to_csv(seq / "radar_candidates.csv", index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    ).to_csv(seq / "truth.csv", index=False)
    discovered = discover_sequence_paths(tmp_path)
    assert len(discovered) == 1
    candidates, truth, calibration = load_sequence_export(discovered[0])
    assert calibration is None
    assert len(candidates.rows) == 1
    assert truth is not None
    assert len(truth.rows) == 1


def test_sequence_loading_fills_missing_sequence_ids_from_sequence_folder(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_without_ids"
    seq.mkdir()
    pd.DataFrame(
        {
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["r1"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    ).to_csv(seq / "candidates.csv", index=False)
    pd.DataFrame(
        {
            "time_s": [1.0, 1.0, 1.0],
            "x": [10.0, 10.1, 10.2],
            "y": [0.0, 0.0, 0.1],
            "z": [1.0, 1.1, 1.0],
        }
    ).to_csv(seq / "points.csv", index=False)
    pd.DataFrame(
        {
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    candidates, truth, _ = load_sequence_export(discover_sequence_paths(tmp_path)[0])

    assert set(candidates.rows["sequence_id"]) == {"seq_without_ids"}
    assert truth is not None
    assert truth.rows["sequence_id"].tolist() == ["seq_without_ids"]


def test_sequence_root_discovers_delimited_candidate_tables(tmp_path: Path) -> None:
    seq = tmp_path / "seq_tsv"
    seq.mkdir()
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "sensor": ["radar", "radar"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [2.0, 2.0],
        }
    ).to_csv(seq / "candidates.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].candidate_csvs == (seq / "candidates.tsv",)
    assert candidates.rows["sequence_id"].tolist() == ["seq_tsv", "seq_tsv"]
    assert candidates.rows["source"].tolist() == ["radar", "radar"]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [0.0, 1.0]


def test_sequence_root_loads_json_candidate_and_truth_exports(tmp_path: Path) -> None:
    seq = tmp_path / "seq_json"
    seq.mkdir()
    (seq / "candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "timestamp_ns": 500_000_000,
                        "sensor": "radar",
                        "x_m": 10.0,
                        "y_m": 1.0,
                        "z_m": 2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (seq / "truth.json").write_text(
        json.dumps(
            {
                "truth": [
                    {
                        "time_s": 0.5,
                        "x_m": 10.0,
                        "y_m": 1.0,
                        "z_m": 2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].candidate_csvs == (seq / "candidates.json",)
    assert discovered[0].truth_files == (seq / "truth.json",)
    assert candidates.rows["sequence_id"].tolist() == ["seq_json"]
    assert candidates.rows["source"].tolist() == ["radar"]
    assert candidates.rows["time_s"].tolist() == [0.5]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [0.5]


def test_sequence_root_loads_jsonl_candidate_truth_and_class_exports(tmp_path: Path) -> None:
    seq = tmp_path / "seq_jsonl"
    seq.mkdir()
    (seq / "candidates.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "timestamp_s": 0.0,
                    "sensor": "radar",
                    "x_m": 10.0,
                    "y_m": 1.0,
                    "z_m": 2.0,
                },
                {
                    "timestamp_s": 1.0,
                    "sensor": "radar",
                    "x_m": 11.0,
                    "y_m": 1.0,
                    "z_m": 2.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    (seq / "truth.ndjson").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"time_s": 0.0, "x_m": 10.0, "y_m": 1.0, "z_m": 2.0},
                {"time_s": 1.0, "x_m": 11.0, "y_m": 1.0, "z_m": 2.0},
            ]
        ),
        encoding="utf-8",
    )
    (seq / "classes.jsonl").write_text(
        json.dumps({"sequence_id": "seq_jsonl", "uav_type": "quadrotor"}),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].candidate_csvs == (seq / "candidates.jsonl",)
    assert discovered[0].truth_files == (seq / "truth.ndjson",)
    assert discovered[0].class_files == (seq / "classes.jsonl",)
    assert candidates.rows["sequence_id"].tolist() == ["seq_jsonl", "seq_jsonl"]
    assert candidates.rows["source"].tolist() == ["radar", "radar"]
    assert candidates.rows["class_name"].tolist() == ["quadrotor", "quadrotor"]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [0.0, 1.0]


def test_sequence_root_loads_gzipped_table_exports(tmp_path: Path) -> None:
    seq = tmp_path / "seq_gzip"
    seq.mkdir()
    pd.DataFrame(
        {
            "timestamp_s": [0.0, 1.0],
            "sensor": ["radar", "radar"],
            "x_m": [10.0, 11.0],
            "y_m": [1.0, 1.0],
            "z_m": [2.0, 2.0],
        }
    ).to_csv(seq / "candidates.csv.gz", index=False, compression="gzip")
    with gzip.open(seq / "truth.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                json.dumps(row)
                for row in [
                    {"time_s": 0.0, "x_m": 10.0, "y_m": 1.0, "z_m": 2.0},
                    {"time_s": 1.0, "x_m": 11.0, "y_m": 1.0, "z_m": 2.0},
                ]
            )
        )
    with gzip.open(seq / "classes.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"sequence_id": "seq_gzip", "uav_type": "quadrotor"}))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].candidate_csvs == (seq / "candidates.csv.gz",)
    assert discovered[0].truth_files == (seq / "truth.jsonl.gz",)
    assert discovered[0].class_files == (seq / "classes.jsonl.gz",)
    assert candidates.rows["sequence_id"].tolist() == ["seq_gzip", "seq_gzip"]
    assert candidates.rows["class_name"].tolist() == ["quadrotor", "quadrotor"]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [0.0, 1.0]


def test_sequence_root_loads_json_class_labels(tmp_path: Path) -> None:
    seq = tmp_path / "seq_json_class"
    seq.mkdir()
    (seq / "candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0},
                    {"time_s": 1.0, "x_m": 1.0, "y_m": 0.0, "z_m": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    (seq / "truth.json").write_text(
        json.dumps(
            {
                "truth": [
                    {"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0},
                    {"time_s": 1.0, "x_m": 1.0, "y_m": 0.0, "z_m": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    (seq / "classes.json").write_text(
        json.dumps({"class_map": {"seq_json_class": {"uav_type": "quadrotor"}}}),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].class_files == (seq / "classes.json",)
    assert candidates.rows["class_name"].tolist() == ["quadrotor", "quadrotor"]
    output = run_mmuad_tracker(candidates, truth)
    results = estimates_to_mmaud_results_frame(output.estimates, class_name="unknown")
    assert results["uav_type"].tolist() == ["quadrotor", "quadrotor"]


def test_sequence_root_loads_yaml_class_labels(tmp_path: Path) -> None:
    seq = tmp_path / "seq_yaml_class"
    seq.mkdir()
    (seq / "candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0},
                    {"time_s": 1.0, "x_m": 1.0, "y_m": 0.0, "z_m": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    (seq / "truth.json").write_text(
        json.dumps(
            {
                "truth": [
                    {"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0},
                    {"time_s": 1.0, "x_m": 1.0, "y_m": 0.0, "z_m": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    (seq / "classes.yaml").write_text(
        "\n".join(
            [
                "class_map:",
                "  seq_yaml_class:",
                "    uav_type: quadrotor",
            ]
        ),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].class_files == (seq / "classes.yaml",)
    assert candidates.rows["class_name"].tolist() == ["quadrotor", "quadrotor"]
    output = run_mmuad_tracker(candidates, truth)
    results = estimates_to_mmaud_results_frame(output.estimates, class_name="unknown")
    assert results["uav_type"].tolist() == ["quadrotor", "quadrotor"]


def test_sequence_root_discovers_delimited_point_tables(tmp_path: Path) -> None:
    seq = tmp_path / "seq_points_tsv"
    seq.mkdir()
    pd.DataFrame(
        {
            "time_s": [3.0, 3.0, 3.0],
            "x": [0.0, 0.1, 0.2],
            "y": [0.0, 0.0, 0.1],
            "z": [2.0, 2.1, 2.0],
        }
    ).to_csv(seq / "lidar_points.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "time_s": [3.0],
            "x_m": [0.1],
            "y_m": [0.0],
            "z_m": [2.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].point_cloud_files == (seq / "lidar_points.tsv",)
    assert len(candidates.rows) == 1
    assert candidates.rows["sequence_id"].tolist() == ["seq_points_tsv"]
    assert abs(float(candidates.rows.loc[0, "time_s"]) - 3.0) < 1e-9
    assert truth is not None


def test_sequence_root_discovers_delimited_radar_and_camera_tables(tmp_path: Path) -> None:
    seq = tmp_path / "seq_modalities_delimited"
    seq.mkdir()
    pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": [10.0],
            "azimuth_deg": [90.0],
        }
    ).to_csv(seq / "radar_polar.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "time_s": [1.0],
            "source": ["cam0"],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [5.0],
        }
    ).to_csv(seq / "camera_detections.tsv", sep="\t", index=False)
    (seq / "calibration.json").write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                },
                "sensors": {},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "x_m": [10.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 5.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].radar_polar_csvs == (seq / "radar_polar.tsv",)
    assert discovered[0].camera_detection_csvs == (seq / "camera_detections.tsv",)
    assert set(candidates.rows["source"]) == {"cam0", "radar_polar"}
    assert truth is not None


def test_sequence_root_discovers_radar_and_camera_modality_folder_tables(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_sensor_folders"
    radar = seq / "radar0"
    camera = seq / "cam0"
    radar.mkdir(parents=True)
    camera.mkdir()
    pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": [10.0],
            "azimuth_deg": [90.0],
        }
    ).to_csv(radar / "detections.csv", index=False)
    pd.DataFrame(
        {
            "time_s": [1.0],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [5.0],
        }
    ).to_csv(camera / "detections.csv", index=False)
    (seq / "calibration.json").write_text(
        json.dumps(
            {
                "sensors": {"radar": {}},
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "x_m": [10.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 5.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0], apply_calibration=False)

    assert [sequence.sequence_id for sequence in discovered] == ["seq_sensor_folders"]
    assert discovered[0].radar_polar_csvs == (radar / "detections.csv",)
    assert discovered[0].camera_detection_csvs == (camera / "detections.csv",)
    assert set(candidates.rows["source"]) == {"cam0", "radar0"}
    assert truth is not None


def test_sequence_root_discovers_json_radar_modality_folder_tables(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_radar_json"
    radar = seq / "radar0"
    radar.mkdir(parents=True)
    (radar / "detections.json").write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "time_s": 0.0,
                        "range_m": 10.0,
                        "azimuth_deg": 90.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (seq / "truth.json").write_text(
        json.dumps(
            [
                {
                    "time_s": 0.0,
                    "x_m": 10.0,
                    "y_m": 0.0,
                    "z_m": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0], apply_calibration=False)

    assert [sequence.sequence_id for sequence in discovered] == ["seq_radar_json"]
    assert discovered[0].radar_polar_csvs == (radar / "detections.json",)
    assert set(candidates.rows["source"]) == {"radar0"}
    assert abs(float(candidates.rows.loc[0, "x_m"]) - 10.0) < 1.0e-9
    assert truth is not None


def test_sequence_root_discovers_cartesian_radar_modality_folder_tables(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_radar_cartesian"
    radar = seq / "radar0"
    radar.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp_ms": [1250],
            "x_m": [2.0],
            "y_m": [3.0],
            "z_m": [4.0],
            "track_id": ["trk-1"],
            "confidence": [0.9],
        }
    ).to_csv(radar / "detections.csv", index=False)
    pd.DataFrame(
        {
            "time_s": [1.25],
            "x_m": [2.0],
            "y_m": [3.0],
            "z_m": [4.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0], apply_calibration=False)

    assert [sequence.sequence_id for sequence in discovered] == ["seq_radar_cartesian"]
    assert discovered[0].candidate_csvs == (radar / "detections.csv",)
    assert discovered[0].radar_polar_csvs == ()
    assert candidates.rows["source"].tolist() == ["radar0"]
    assert abs(float(candidates.rows.loc[0, "time_s"]) - 1.25) < 1.0e-9
    assert abs(float(candidates.rows.loc[0, "x_m"]) - 2.0) < 1.0e-9
    assert truth is not None


def test_sequence_root_discovers_json_cartesian_radar_modality_folder_tables(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_radar_cartesian_json"
    radar = seq / "mmwave"
    radar.mkdir(parents=True)
    (radar / "detections.json").write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "timestamp": 1.25,
                        "x": 2.0,
                        "y": 3.0,
                        "z": 4.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (seq / "truth.json").write_text(
        json.dumps(
            [
                {
                    "time_s": 1.25,
                    "x_m": 2.0,
                    "y_m": 3.0,
                    "z_m": 4.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0], apply_calibration=False)

    assert [sequence.sequence_id for sequence in discovered] == [
        "seq_radar_cartesian_json"
    ]
    assert discovered[0].candidate_csvs == (radar / "detections.json",)
    assert discovered[0].radar_polar_csvs == ()
    assert candidates.rows["source"].tolist() == ["mmwave"]
    assert abs(float(candidates.rows.loc[0, "time_s"]) - 1.25) < 1.0e-9
    assert abs(float(candidates.rows.loc[0, "y_m"]) - 3.0) < 1.0e-9
    assert truth is not None


def test_sequence_root_discovers_json_camera_detection_tables(tmp_path: Path) -> None:
    seq = tmp_path / "seq_camera_json"
    camera = seq / "cam0"
    camera.mkdir(parents=True)
    (camera / "detections.json").write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "timestamp_ms": 1250,
                        "u_px": 50.0,
                        "v_px": 50.0,
                        "depth_m": 5.0,
                        "score": 0.8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (seq / "calibration.json").write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (seq / "truth.json").write_text(
        json.dumps({"truth": [{"time_s": 1.25, "x_m": 0.0, "y_m": 0.0, "z_m": 5.0}]}),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].camera_detection_csvs == (camera / "detections.json",)
    row = candidates.rows.iloc[0]
    assert row["source"] == "cam0"
    assert abs(float(row["time_s"]) - 1.25) < 1e-12
    assert abs(float(row["z_m"]) - 5.0) < 1e-12
    assert truth is not None


def test_sequence_root_discovers_camera_compact_bbox_columns(tmp_path: Path) -> None:
    seq = tmp_path / "seq_camera_bbox"
    camera = seq / "cam0"
    camera.mkdir(parents=True)
    pd.DataFrame(
        {
            "time_s": [0.0],
            "bbox": ["[40, 40, 20, 20]"],
            "depth_m": [5.0],
        }
    ).to_csv(camera / "export.csv", index=False)
    (seq / "calibration.json").write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [5.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].camera_detection_csvs == (camera / "export.csv",)
    row = candidates.rows.iloc[0]
    assert row["source"] == "cam0"
    assert (row["x_m"], row["y_m"], row["z_m"]) == (0.0, 0.0, 5.0)
    assert truth is not None


def test_sequence_root_discovers_camera_info_intrinsics_file(tmp_path: Path) -> None:
    seq = tmp_path / "seq_camera_info"
    camera = seq / "cam0"
    camera.mkdir(parents=True)
    pd.DataFrame(
        {
            "time_s": [0.0],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [5.0],
        }
    ).to_csv(camera / "detections.csv", index=False)
    (seq / "camera_info.json").write_text(
        json.dumps(
            {
                "source": "cam0",
                "width": 100,
                "height": 100,
                "k": [
                    100.0,
                    0.0,
                    50.0,
                    0.0,
                    100.0,
                    50.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [5.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, calibration = load_sequence_export(discovered[0])

    assert discovered[0].calibration_file == seq / "camera_info.json"
    assert discovered[0].camera_calibration_files == (seq / "camera_info.json",)
    assert discovered[0].camera_detection_csvs == (camera / "detections.csv",)
    row = candidates.rows.iloc[0]
    assert row["source"] == "cam0"
    assert abs(float(row["x_m"])) < 1.0e-12
    assert abs(float(row["y_m"])) < 1.0e-12
    assert abs(float(row["z_m"]) - 5.0) < 1.0e-12
    assert truth is not None
    assert calibration is None


def test_sequence_root_discovers_camera_folder_intrinsics_file(tmp_path: Path) -> None:
    seq = tmp_path / "seq_camera_folder_info"
    camera = seq / "cam0"
    camera.mkdir(parents=True)
    pd.DataFrame(
        {
            "time_s": [0.0],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [5.0],
        }
    ).to_csv(camera / "detections.csv", index=False)
    (camera / "camera_info.json").write_text(
        json.dumps(
            {
                "width": 100,
                "height": 100,
                "k": [
                    100.0,
                    0.0,
                    50.0,
                    0.0,
                    100.0,
                    50.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [5.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, calibration = load_sequence_export(discovered[0])

    assert discovered[0].calibration_file is None
    assert discovered[0].camera_calibration_files == (camera / "camera_info.json",)
    assert discovered[0].camera_detection_csvs == (camera / "detections.csv",)
    row = candidates.rows.iloc[0]
    assert row["source"] == "cam0"
    assert abs(float(row["x_m"])) < 1.0e-12
    assert abs(float(row["y_m"])) < 1.0e-12
    assert abs(float(row["z_m"]) - 5.0) < 1.0e-12
    assert truth is not None
    assert calibration is None


def test_sequence_root_discovers_camera_folder_yaml_intrinsics_file(tmp_path: Path) -> None:
    seq = tmp_path / "seq_camera_folder_yaml_info"
    camera = seq / "cam0"
    camera.mkdir(parents=True)
    pd.DataFrame(
        {
            "time_s": [0.0],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [5.0],
        }
    ).to_csv(camera / "detections.csv", index=False)
    (camera / "camera_info.yaml").write_text(
        json.dumps(
            {
                "width": 100,
                "height": 100,
                "k": [
                    100.0,
                    0.0,
                    50.0,
                    0.0,
                    100.0,
                    50.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [5.0],
        }
    ).to_csv(seq / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, calibration = load_sequence_export(discovered[0])

    assert discovered[0].calibration_file is None
    assert discovered[0].camera_calibration_files == (camera / "camera_info.yaml",)
    assert discovered[0].camera_detection_csvs == (camera / "detections.csv",)
    row = candidates.rows.iloc[0]
    assert row["source"] == "cam0"
    assert abs(float(row["x_m"])) < 1.0e-12
    assert abs(float(row["y_m"])) < 1.0e-12
    assert abs(float(row["z_m"]) - 5.0) < 1.0e-12
    assert truth is not None
    assert calibration is None


def test_sequence_root_loads_exported_topic_map_sequence(tmp_path: Path) -> None:
    seq = tmp_path / "seq_topic_map"
    seq.mkdir()
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [4.0, 4.0],
        }
    ).to_csv(seq / "radar_export.csv", index=False)
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [4.0, 4.0],
        }
    ).to_csv(seq / "truth_export.csv", index=False)
    topic_map = seq / "topic_map.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_topic_map",
                "exports": [
                    {
                        "kind": "candidate",
                        "path": "radar_export.csv",
                        "source": "radar",
                        "column_aliases": {
                            "stamp": "time_s",
                            "px": "x_m",
                            "py": "y_m",
                            "pz": "z_m",
                        },
                    },
                    {
                        "kind": "pose_truth",
                        "path": "truth_export.csv",
                        "column_aliases": {
                            "stamp": "time_s",
                            "px": "x_m",
                            "py": "y_m",
                            "pz": "z_m",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].topic_map_jsons == (topic_map,)
    assert candidates.rows["source"].tolist() == ["radar", "radar"]
    assert candidates.rows["sequence_id"].tolist() == ["seq_topic_map", "seq_topic_map"]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [0.0, 1.0]


def test_sequence_root_loads_exported_yaml_topic_map_sequence(tmp_path: Path) -> None:
    seq = tmp_path / "seq_topic_map_yaml"
    seq.mkdir()
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [4.0, 4.0],
        }
    ).to_csv(seq / "radar_export.csv", index=False)
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [4.0, 4.0],
        }
    ).to_csv(seq / "truth_export.csv", index=False)
    topic_map = seq / "topic_map.yaml"
    topic_map.write_text(
        "\n".join(
            [
                "sequence_id: seq_topic_map_yaml",
                "exports:",
                "  - kind: candidate",
                "    path: radar_export.csv",
                "    source: radar",
                "    column_aliases:",
                "      stamp: time_s",
                "      px: x_m",
                "      py: y_m",
                "      pz: z_m",
                "  - kind: pose_truth",
                "    path: truth_export.csv",
                "    column_aliases:",
                "      stamp: time_s",
                "      px: x_m",
                "      py: y_m",
                "      pz: z_m",
            ]
        ),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].topic_map_jsons == (topic_map,)
    assert candidates.rows["source"].tolist() == ["radar", "radar"]
    assert candidates.rows["sequence_id"].tolist() == [
        "seq_topic_map_yaml",
        "seq_topic_map_yaml",
    ]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [0.0, 1.0]


def test_sequence_root_skips_native_only_topic_map_template(tmp_path: Path) -> None:
    seq = tmp_path / "seq_native_only"
    seq.mkdir()
    (seq / "topic_map_native.json").write_text(
        json.dumps(
            {
                "sequence_id": "seq_native_only",
                "exports": [
                    {
                        "topic": "/radar/points",
                        "kind": "pointcloud2_candidate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert discover_sequence_paths(tmp_path) == []


def test_sequence_root_discovers_native_topic_map_with_recording(tmp_path: Path) -> None:
    seq = tmp_path / "seq_native_recording"
    seq.mkdir()
    bag = seq / "recording.mcap"
    bag.write_bytes(b"fake native bag")
    topic_map = seq / "topic_map_native.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_native_recording",
                "exports": [
                    {
                        "topic": "/radar/points",
                        "kind": "pointcloud2_candidate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)

    assert len(discovered) == 1
    assert discovered[0].sequence_id == "seq_native_recording"
    assert discovered[0].native_topic_map_jsons == (topic_map,)
    assert discovered[0].rosbag_paths == (bag,)
    assert discovered[0].topic_map_jsons == ()


def test_sequence_root_discovers_native_ros2_bag_directory(tmp_path: Path) -> None:
    seq = tmp_path / "seq_native_ros2"
    seq.mkdir()
    (seq / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  relative_file_paths:",
                "    - data_0.db3",
            ]
        ),
        encoding="utf-8",
    )
    (seq / "data_0.db3").write_bytes(b"fake sqlite bag")
    topic_map = seq / "topic_map_native.yaml"
    topic_map.write_text(
        "\n".join(
            [
                "sequence_id: seq_native_ros2",
                "exports:",
                "  - topic: /radar/points",
                "    kind: pointcloud2_candidate",
            ]
        ),
        encoding="utf-8",
    )

    discovered = discover_sequence_paths(tmp_path)

    assert len(discovered) == 1
    assert discovered[0].native_topic_map_jsons == (topic_map,)
    assert discovered[0].rosbag_paths == (seq,)


def test_cli_sequence_root_runs_topic_map_only_sequence(tmp_path: Path) -> None:
    seq = tmp_path / "seq_topic_map_cli"
    seq.mkdir()
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [4.0, 4.0, 4.0],
        }
    ).to_csv(seq / "detections.csv", index=False)
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [4.0, 4.0, 4.0],
        }
    ).to_csv(seq / "truth_export.csv", index=False)
    (seq / "topic_map.json").write_text(
        json.dumps(
            {
                "sequence_id": "seq_topic_map_cli",
                "exports": [
                    {
                        "kind": "odometry_candidate",
                        "path": "detections.csv",
                        "source": "odom",
                        "column_aliases": {
                            "stamp": "time_s",
                            "x": "x_m",
                            "y": "y_m",
                            "z": "z_m",
                        },
                    },
                    {
                        "kind": "odometry_truth",
                        "path": "truth_export.csv",
                        "column_aliases": {
                            "stamp": "time_s",
                            "x": "x_m",
                            "y": "y_m",
                            "z": "z_m",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--sequence-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--submission-csv",
            str(output / "submission.csv"),
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output / "mmuad_estimates.csv")
    assert estimates["sequence_id"].tolist() == ["seq_topic_map_cli"] * 3
    assert (output / "submission.csv").exists()


def test_submission_writers_use_estimate_state_columns(tmp_path: Path) -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 1.0],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
        }
    )
    frame = estimates_to_submission_frame(estimates, track_id="uav0")
    assert list(frame.columns) == [
        "sequence_id",
        "time_s",
        "track_id",
        "x_m",
        "y_m",
        "z_m",
        "score",
    ]
    assert frame.loc[1, "track_id"] == "uav0"
    json_path = write_submission_json(estimates, tmp_path / "submission.json", track_id="uav0")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "raft-uav-mmuad-single-uav-trajectory-v1"
    assert len(payload["sequences"]["s1"]) == 2


def test_submission_frame_drops_nonfinite_estimates() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1"],
            "time_s": [0.0, 1.0, np.nan],
            "output_track_id": ["mot_1", None, "mot_bad"],
            "state_x_m": [1.0, np.nan, 3.0],
            "state_y_m": [2.0, 2.0, 2.0],
            "state_z_m": [3.0, 3.0, 3.0],
        }
    )

    frame = estimates_to_submission_frame(estimates, track_id="fallback")

    assert len(frame) == 1
    assert frame.loc[0, "track_id"] == "mot_1"
    assert np.isfinite(frame[["time_s", "x_m", "y_m", "z_m"]].to_numpy(dtype=float)).all()


def test_submission_frame_fills_missing_estimate_track_ids() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "time_s": [0.0],
            "output_track_id": [None],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    )

    frame = estimates_to_submission_frame(estimates, track_id="fallback")

    assert frame.loc[0, "track_id"] == "fallback"


def test_trajectory_metrics_use_latest_time_for_fde() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1"],
            "time_s": [2.0, 0.0, 1.0],
            "error_3d_m": [20.0, 0.0, 10.0],
            "error_2d_m": [12.0, 0.0, 6.0],
        }
    )

    metrics = compute_trajectory_metrics(estimates)

    assert metrics["pooled"]["fde_3d_m"] == 20.0
    assert metrics["pooled"]["fde_2d_m"] == 12.0
    assert metrics["sequences"]["s1"]["fde_3d_m"] == 20.0


def _minimal_las_bytes(points: list[tuple[float, float, float]]) -> bytes:
    import struct

    scale = (0.01, 0.01, 0.01)
    offset = (0.0, 0.0, 0.0)
    header_size = 227
    point_record_length = 20
    header = bytearray(header_size)
    header[0:4] = b"LASF"
    header[24] = 1
    header[25] = 2
    struct.pack_into("<H", header, 94, header_size)
    struct.pack_into("<I", header, 96, header_size)
    struct.pack_into("<B", header, 104, 0)
    struct.pack_into("<H", header, 105, point_record_length)
    struct.pack_into("<I", header, 107, len(points))
    struct.pack_into("<ddd", header, 131, *scale)
    struct.pack_into("<ddd", header, 155, *offset)
    payload = bytearray()
    for x_m, y_m, z_m in points:
        x_raw = int(round((x_m - offset[0]) / scale[0]))
        y_raw = int(round((y_m - offset[1]) / scale[1]))
        z_raw = int(round((z_m - offset[2]) / scale[2]))
        payload.extend(struct.pack("<iii", x_raw, y_raw, z_raw))
        payload.extend(b"\x00" * (point_record_length - 12))
    return bytes(header) + bytes(payload)


def _lzf_literal_payload(payload: bytes) -> bytes:
    encoded = bytearray()
    for start in range(0, len(payload), 32):
        chunk = payload[start : start + 32]
        encoded.append(len(chunk) - 1)
        encoded.extend(chunk)
    return bytes(encoded)


def test_ascii_pcd_point_cloud_is_clustered(tmp_path: Path) -> None:
    pcd = tmp_path / "frame_12.5.pcd"
    pcd.write_text(
        "\n".join(
            [
                "# .PCD v0.7",
                "VERSION 0.7",
                "FIELDS x y z",
                "SIZE 4 4 4",
                "TYPE F F F",
                "COUNT 1 1 1",
                "WIDTH 3",
                "HEIGHT 1",
                "POINTS 3",
                "DATA ascii",
                "0 0 1",
                "0.1 0 1.1",
                "0.2 0.1 1.0",
            ]
        ),
        encoding="utf-8",
    )
    frame = load_point_cloud_file_as_candidates(pcd, voxel_size_m=0.5, min_points=3)
    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 12.5) < 1e-9


def test_gzipped_ascii_pcd_point_cloud_is_clustered(tmp_path: Path) -> None:
    pcd = tmp_path / "frame_12.5.pcd.gz"
    text = "\n".join(
        [
            "# .PCD v0.7",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            "WIDTH 3",
            "HEIGHT 1",
            "POINTS 3",
            "DATA ascii",
            "0 0 1",
            "0.1 0 1.1",
            "0.2 0.1 1.0",
        ]
    )
    with gzip.open(pcd, "wt", encoding="utf-8") as handle:
        handle.write(text)

    frame = load_point_cloud_file_as_candidates(pcd, voxel_size_m=0.5, min_points=3)

    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 12.5) < 1e-9


def test_csv_point_cloud_file_infers_missing_metadata(tmp_path: Path) -> None:
    csv = tmp_path / "frame_12.5.csv"
    pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2],
            "y": [0.0, 0.0, 0.1],
            "z": [1.0, 1.1, 1.0],
        }
    ).to_csv(csv, index=False)

    frame = load_point_cloud_file_as_candidates(
        csv, sequence_id="seq_csv", voxel_size_m=0.5, min_points=3
    )

    assert len(frame.rows) == 1
    assert frame.rows.loc[0, "sequence_id"] == "seq_csv"
    assert abs(float(frame.rows.loc[0, "time_s"]) - 12.5) < 1e-9


def test_tsv_point_cloud_file_is_clustered(tmp_path: Path) -> None:
    tsv = tmp_path / "livox_points_4.5.tsv"
    pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2],
            "y": [0.0, 0.0, 0.1],
            "z": [1.0, 1.1, 1.0],
        }
    ).to_csv(tsv, sep="\t", index=False)

    frame = load_point_cloud_file_as_candidates(
        tsv,
        sequence_id="seq_livox",
        voxel_size_m=0.5,
        min_points=3,
    )

    assert len(frame.rows) == 1
    assert frame.rows.loc[0, "sequence_id"] == "seq_livox"
    assert abs(float(frame.rows.loc[0, "time_s"]) - 4.5) < 1e-9


def test_json_point_cloud_file_is_clustered(tmp_path: Path) -> None:
    points = tmp_path / "livox_points_4.5.json"
    points.write_text(
        json.dumps(
            {
                "points": [
                    {"x": 0.0, "y": 0.0, "z": 1.0},
                    {"x": 0.1, "y": 0.0, "z": 1.1},
                    {"x": 0.2, "y": 0.1, "z": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    frame = load_point_cloud_file_as_candidates(
        points,
        sequence_id="seq_livox_json",
        voxel_size_m=0.5,
        min_points=3,
    )

    assert len(frame.rows) == 1
    assert frame.rows.loc[0, "sequence_id"] == "seq_livox_json"
    assert abs(float(frame.rows.loc[0, "time_s"]) - 4.5) < 1e-9


def test_split_manifest_filters_discovered_sequences(tmp_path: Path) -> None:
    for name in ("seq_train", "seq_val"):
        seq = tmp_path / name
        seq.mkdir()
        pd.DataFrame(
            {
                "sequence_id": [name],
                "time_s": [0.0],
                "source": ["radar"],
                "track_id": ["r1"],
                "x_m": [0.0],
                "y_m": [0.0],
                "z_m": [1.0],
            }
        ).to_csv(seq / "candidates.csv", index=False)
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"train": ["seq_train"], "val": ["seq_val"]}))
    manifest = load_split_manifest(split)
    discovered = discover_sequence_paths(tmp_path)
    val_sequences = filter_sequences_by_split(discovered, manifest, "val")
    assert [sequence.sequence_id for sequence in val_sequences] == ["seq_val"]


def test_split_manifest_filters_relative_sequence_paths(tmp_path: Path) -> None:
    for split, x_m in (("train", 100.0), ("val", 1.0)):
        seq = tmp_path / split / "seq_same"
        seq.mkdir(parents=True)
        pd.DataFrame(
            {
                "sequence_id": ["seq_same"],
                "time_s": [0.0],
                "source": ["radar"],
                "track_id": ["r1"],
                "x_m": [x_m],
                "y_m": [0.0],
                "z_m": [1.0],
            }
        ).to_csv(seq / "candidates.csv", index=False)
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps(
            {
                "train": ["train/seq_same"],
                "val": ["val/seq_same"],
                "val_windows": ["val\\seq_same"],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_split_manifest(split)
    discovered = discover_sequence_paths(tmp_path)

    val_sequences = filter_sequences_by_split(discovered, manifest, "val")
    val_windows_sequences = filter_sequences_by_split(discovered, manifest, "val_windows")

    assert [sequence.root.parent.name for sequence in val_sequences] == ["val"]
    assert [sequence.root.parent.name for sequence in val_windows_sequences] == ["val"]


def test_split_manifest_accepts_nested_json_layouts(tmp_path: Path) -> None:
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "sequences": [
                            {"sequence_id": "seq_train"},
                            {"id": "seq_train_2"},
                        ]
                    },
                    "val": {"sequence_ids": ["seq_val"]},
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(split)

    assert manifest["train"] == ("seq_train", "seq_train_2")
    assert manifest["val"] == ("seq_val",)


def test_split_manifest_accepts_yaml_layouts(tmp_path: Path) -> None:
    split = tmp_path / "splits.yaml"
    split.write_text(
        "\n".join(
            [
                "splits:",
                "  train:",
                "    sequences:",
                "      - sequence_id: train/seq_train",
                "      - id: train/seq_train_2",
                "  val:",
                "    sequence_ids:",
                "      - val/seq_val",
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(split)

    assert manifest["train"] == ("train/seq_train", "train/seq_train_2")
    assert manifest["val"] == ("val/seq_val",)


def test_split_manifest_accepts_sequence_rows_json(tmp_path: Path) -> None:
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps(
            {
                "sequences": [
                    {"name": "seq_train", "subset": "train"},
                    {"id": "seq_val", "partition": "val"},
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(split)

    assert manifest["train"] == ("seq_train",)
    assert manifest["val"] == ("seq_val",)


def test_split_manifest_accepts_csv_alias_columns(tmp_path: Path) -> None:
    split = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "id": ["seq_train", "seq_val"],
            "subset": ["train", "val"],
        }
    ).to_csv(split, index=False)

    manifest = load_split_manifest(split)

    assert manifest["train"] == ("seq_train",)
    assert manifest["val"] == ("seq_val",)


def test_cli_split_name_filters_top_level_split_folders_without_manifest(tmp_path: Path) -> None:
    for split, name, x_m in (("train", "seq_train", 100.0), ("val", "seq_val", 1.0)):
        seq = tmp_path / split / name
        seq.mkdir(parents=True)
        pd.DataFrame(
            {
                "time_s": [0.0, 1.0],
                "source": ["radar", "radar"],
                "track_id": ["r1", "r1"],
                "x_m": [x_m, x_m + 1.0],
                "y_m": [0.0, 0.0],
                "z_m": [1.0, 1.0],
            }
        ).to_csv(seq / "candidates.csv", index=False)
        pd.DataFrame(
            {
                "time_s": [0.0, 1.0],
                "x_m": [x_m, x_m + 1.0],
                "y_m": [0.0, 0.0],
                "z_m": [1.0, 1.0],
            }
        ).to_csv(seq / "truth.csv", index=False)
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--sequence-root",
            str(tmp_path),
            "--split-name",
            "val",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output / "mmuad_estimates.csv")
    assert set(estimates["sequence_id"]) == {"seq_val"}
    assert estimates["state_x_m"].max() < 10.0


def test_multi_object_tracker_outputs_tracks_and_mot_metrics(tmp_path: Path) -> None:
    cand_rows = []
    truth_rows = []
    for i in range(4):
        t = float(i)
        for object_id, y in (("a", 0.0), ("b", 10.0)):
            truth_rows.append(
                {
                    "sequence_id": "s1",
                    "time_s": t,
                    "object_id": object_id,
                    "x_m": t,
                    "y_m": y,
                    "z_m": 1.0,
                }
            )
            cand_rows.append(
                {
                    "sequence_id": "s1",
                    "time_s": t,
                    "source": "lidar",
                    "track_id": object_id,
                    "x_m": t,
                    "y_m": y,
                    "z_m": 1.0,
                    "confidence": 0.9,
                }
            )
    cand_path = tmp_path / "mot_candidates.csv"
    truth_path = tmp_path / "mot_truth.csv"
    pd.DataFrame(cand_rows).to_csv(cand_path, index=False)
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    output = run_mmuad_multi_object_tracker(
        load_candidate_csv(cand_path),
        load_truth_csv(truth_path),
        config=MultiObjectTrackerConfig(max_association_distance_m=3.0),
    )
    assert output.estimates["output_track_id"].nunique() == 2
    assert output.metrics["pooled"]["matches"] == 8
    assert output.metrics["pooled"]["id_switches"] == 0


def test_multi_object_tracker_ignores_invalid_manual_detections() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1", "s1"],
                "time_s": [0.0, 1.0, 1.0],
                "source": ["radar", "bad", "radar"],
                "track_id": [None, None, None],
                "x_m": [0.0, np.nan, 1.0],
                "y_m": [0.0, 0.0, 0.0],
                "z_m": [2.0, 2.0, 2.0],
                "confidence": [1.0, 1.0, 0.1],
                "std_xy_m": [10.0, np.nan, 10.0],
                "std_z_m": [10.0, np.nan, 10.0],
            }
        )
    )

    output = run_mmuad_multi_object_tracker(candidates, config=MultiObjectTrackerConfig())

    assert output.estimates["source"].tolist() == ["radar", "radar"]
    assert np.isfinite(
        output.estimates[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(dtype=float)
    ).all()


def test_multi_object_metrics_ignore_invalid_truth_and_estimates() -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "output_track_id": ["mot_1", "mot_bad"],
            "state_x_m": [0.0, np.nan],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [2.0, 2.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "track_id": ["uav", "bad_truth"],
            "x_m": [0.0, np.nan],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["count"] == 1
    assert metrics["gt_count"] == 1
    assert metrics["matches"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_multi_object_metrics_do_not_match_across_sequences() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq_a", "seq_b"],
            "time_s": [0.0, 0.0],
            "output_track_id": ["mot_a", "mot_b"],
            "state_x_m": [100.0, 0.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [2.0, 2.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq_a", "seq_b"],
            "time_s": [0.0, 0.0],
            "track_id": ["uav_a", "uav_b"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=5.0)

    assert metrics["matches"] == 0
    assert metrics["false_positive"] == 2
    assert metrics["false_negative"] == 2
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_multi_object_metrics_without_matches_are_strict_json_compatible() -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [0.0],
            "output_track_id": ["mot_1"],
            "state_x_m": [100.0],
            "state_y_m": [0.0],
            "state_z_m": [2.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "track_id": ["uav"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [2.0],
        }
    )

    metrics = compute_multi_object_metrics(estimates, truth, match_distance_m=1.0)

    assert metrics["matches"] == 0
    assert metrics["motp_3d_m"] is None
    json.dumps(metrics, allow_nan=False)


def test_submission_zip_preserves_multi_object_track_ids(tmp_path: Path) -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 1.0],
            "output_track_id": ["mot_1", "mot_2"],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
        }
    )
    zip_path = write_submission_zip(estimates, tmp_path / "submission.zip")
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert {"submission.csv", "submission.json"}.issubset(names)
        csv_text = archive.read("submission.csv").decode("utf-8")
    assert "mot_1" in csv_text
    assert "mot_2" in csv_text


def test_ug2_codabench_zip_contains_mmaud_results_csv(tmp_path: Path) -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 1.0],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
        }
    )
    csv_path = write_mmaud_results_csv(
        estimates, tmp_path / "mmaud_results.csv", class_name="Mavic3"
    )
    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == [
        "sequence_id",
        "timestamp",
        "x",
        "y",
        "z",
        "uav_type",
        "score",
    ]
    assert frame.loc[0, "uav_type"] == "Mavic3"

    zip_path = write_ug2_codabench_zip(
        estimates, tmp_path / "codabench.zip", class_name="Mavic3"
    )
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    summary = inspect_submission_zip(zip_path)
    assert summary["has_mmaud_results_csv"]
    assert summary["row_count"] == 2
    loaded = load_mmaud_results_file(zip_path)
    assert loaded.rows["uav_type"].tolist() == ["Mavic3", "Mavic3"]


def test_mmaud_results_accept_nanosecond_timestamps() -> None:
    frame = validate_mmaud_results_frame(
        pd.DataFrame(
            {
                "sequence_id": ["seq1"],
                "timestamp_ns": [2_250_000_000],
                "x": [1.0],
                "y": [2.0],
                "z": [3.0],
                "uav_type": ["Mavic3"],
                "score": [1.0],
            }
        )
    )

    assert abs(float(frame.loc[0, "timestamp"]) - 2.25) < 1e-12


def test_local_evaluator_reports_pose_mse_and_type_accuracy(tmp_path: Path) -> None:
    results = tmp_path / "mmaud_results.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "timestamp": [0.0, 1.0],
            "x": [0.0, 2.0],
            "y": [0.0, 0.0],
            "z": [10.0, 10.0],
            "uav_type": ["Mavic3", "Mavic3"],
            "score": [1.0, 1.0],
        }
    ).to_csv(results, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "uav_type": ["Mavic3", "Mavic3"],
        }
    ).to_csv(truth, index=False)

    evaluated = evaluate_mmaud_results(
        load_mmaud_results_csv(results),
        load_truth_csv(truth),
    )

    pooled = evaluated["summary"]["pooled"]
    assert pooled["pose_mse_loss_m2"] == 0.5
    assert pooled["uav_type_accuracy"] == 1.0


def test_sequence_class_map_overrides_submission_type(tmp_path: Path) -> None:
    class_map_csv = tmp_path / "classes.csv"
    class_map_csv.write_text(
        "sequence_id,uav_type\nseqA,Mavic3\nseqB,Phantom4\n",
        encoding="utf-8",
    )
    mapping = load_sequence_class_map(class_map_csv)
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
        }
    )

    frame = estimates_to_mmaud_results_frame(
        estimates,
        class_name="unknown",
        class_map=mapping,
    )

    assert list(frame["uav_type"]) == ["Mavic3", "Phantom4"]


def test_sequence_class_map_accepts_json_row_layouts(tmp_path: Path) -> None:
    class_map_json = tmp_path / "classes.json"
    class_map_json.write_text(
        json.dumps(
            {
                "schema": "exported-class-map-v1",
                "sequences": [
                    {"id": "seqA", "type": "Mavic3"},
                    {"name": "seqB", "category": "Phantom4"},
                ],
            }
        ),
        encoding="utf-8",
    )

    mapping = load_sequence_class_map(class_map_json)

    assert mapping == {"seqA": "Mavic3", "seqB": "Phantom4"}


def test_sequence_class_map_accepts_nested_json_mapping(tmp_path: Path) -> None:
    class_map_json = tmp_path / "classes.json"
    class_map_json.write_text(
        json.dumps(
            {
                "class_map": {
                    "seqA": {"uav_type": "Mavic3"},
                    "seqB": {"label": "Phantom4"},
                }
            }
        ),
        encoding="utf-8",
    )

    mapping = load_sequence_class_map(class_map_json)

    assert mapping == {"seqA": "Mavic3", "seqB": "Phantom4"}


def test_sequence_class_map_accepts_yaml_mapping(tmp_path: Path) -> None:
    class_map_yaml = tmp_path / "classes.yaml"
    class_map_yaml.write_text(
        "\n".join(
            [
                "class_map:",
                "  seqA:",
                "    uav_type: Mavic3",
                "  seqB:",
                "    label: Phantom4",
            ]
        ),
        encoding="utf-8",
    )

    mapping = load_sequence_class_map(class_map_yaml)

    assert mapping == {"seqA": "Mavic3", "seqB": "Phantom4"}


def test_sequence_class_map_accepts_csv_alias_columns(tmp_path: Path) -> None:
    class_map_csv = tmp_path / "classes.csv"
    class_map_csv.write_text("id,type\nseqA,Mavic3\nseqB,Phantom4\n", encoding="utf-8")

    mapping = load_sequence_class_map(class_map_csv)

    assert mapping == {"seqA": "Mavic3", "seqB": "Phantom4"}


def test_cli_writes_ug2_results_with_class_map_file_alias(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    truth = tmp_path / "truth.csv"
    class_map = tmp_path / "classes.yaml"
    output = tmp_path / "out"
    results = output / "mmaud_results.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "source": ["radar", "radar"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
        }
    ).to_csv(candidates, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
        }
    ).to_csv(truth, index=False)
    class_map.write_text(
        "\n".join(["class_map:", "  seqA:", "    uav_type: Mavic3"]),
        encoding="utf-8",
    )

    status = mmuad_cli_main(
        [
            "--candidate-csv",
            str(candidates),
            "--truth-csv",
            str(truth),
            "--ug2-class-map-file",
            str(class_map),
            "--ug2-results-csv",
            str(results),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    rows = pd.read_csv(results)
    assert rows["sequence_id"].tolist() == ["seqA", "seqA"]
    assert rows["uav_type"].tolist() == ["Mavic3", "Mavic3"]


def test_cli_writes_official_track5_results_and_zip(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    truth = tmp_path / "truth.csv"
    class_map = tmp_path / "classes.yaml"
    output = tmp_path / "out"
    results = output / "official_mmaud_results.csv"
    zip_path = output / "official_submission.zip"
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "source": ["radar", "radar"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
        }
    ).to_csv(candidates, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
        }
    ).to_csv(truth, index=False)
    class_map.write_text(
        "\n".join(["class_map:", "  seqA:", "    uav_type: 2"]),
        encoding="utf-8",
    )

    status = mmuad_cli_main(
        [
            "--candidate-csv",
            str(candidates),
            "--truth-csv",
            str(truth),
            "--ug2-class-map-file",
            str(class_map),
            "--ug2-official-results-csv",
            str(results),
            "--ug2-official-codabench-zip",
            str(zip_path),
            "--ug2-official-validate-on-write",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    rows = pd.read_csv(results)
    assert rows.columns.tolist() == [
        "Sequence",
        "Timestamp",
        "Position",
        "Classification",
    ]
    assert rows["Classification"].tolist() == [2, 2]
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    loaded = load_mmaud_results_file(zip_path)
    assert loaded.rows["uav_type"].tolist() == ["2", "2"]
    validation = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["valid"] is True
    assert validation["template_checked"] is True
    assert validation["template_timestamp_count"] == 2
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    manifest = json.loads(
        (output / "mmuad_official_upload_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema"] == "raft-uav-mmuad-official-upload-manifest-v1"
    assert manifest["artifact_path"] == str(zip_path)
    assert manifest["validation_json"] == str(
        output / "mmuad_official_submission_validation.json"
    )
    assert manifest["validation_rows_csv"] == str(
        output / "mmuad_official_submission_validation_rows.csv"
    )
    assert manifest["codabench_upload_ready"] is True
    assert manifest["leaderboard_ready"] is True
    assert manifest["score_valid_for_leaderboard"] is True
    assert manifest["valid"] is True
    assert manifest["sequence_count"] == 1
    assert manifest["ready_sequence_count"] == 1
    assert manifest["blocking_sequence_count"] == 0
    assert manifest["blocking_sequences"] == []
    assert manifest["sequences"]["seqA"]["leaderboard_ready"] is True
    assert manifest["sequences"]["seqA"]["template_timestamp_count"] == 2


def test_cli_validate_on_write_requires_truth_or_template_for_official_readiness(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates.csv"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["radar"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [10.0],
            "class_name": [2],
        }
    ).to_csv(candidates, index=False)

    with pytest.raises(SystemExit, match="timestamp_template_not_checked"):
        mmuad_cli_main(
            [
                "--candidate-csv",
                str(candidates),
                "--ug2-official-codabench-zip",
                str(output / "official_submission.zip"),
                "--ug2-official-classification",
                "2",
                "--ug2-official-validate-on-write",
                "--output-dir",
                str(output),
            ]
        )

    validation = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["valid"] is True
    assert validation["template_checked"] is False
    assert validation["leaderboard_ready"] is False
    assert validation["codabench_upload_ready"] is False
    assert validation["leaderboard_blocking_reasons"] == [
        "timestamp_template_not_checked"
    ]
    manifest = json.loads(
        (output / "mmuad_official_upload_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["codabench_upload_ready"] is False
    assert manifest["leaderboard_ready"] is False
    assert manifest["blocking_sequences"] == ["seqA"]


def test_results_completion_resamples_to_truth_timestamps(tmp_path: Path) -> None:
    results = validate_mmaud_results_frame(
        pd.DataFrame(
            {
                "sequence_id": ["seq1", "seq1"],
                "timestamp": [0.0, 2.0],
                "x": [0.0, 2.0],
                "y": [0.0, 0.0],
                "z": [10.0, 10.0],
                "uav_type": ["Mavic3", "Mavic3"],
                "score": [1.0, 0.5],
            }
        )
    )
    truth_path = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [10.0, 10.0, 10.0],
        }
    ).to_csv(truth_path, index=False)

    completed = complete_results_to_truth_timestamps(
        results,
        load_truth_csv(truth_path),
        max_interpolation_gap_s=3.0,
    )

    assert len(completed.rows) == 3
    middle = completed.rows.loc[completed.rows["timestamp"] == 1.0].iloc[0]
    assert middle["x"] == 1.0
    assert set(completed.diagnostics["completion_method"]) == {"exact", "interpolated"}


def test_layout_inspector_classifies_realistic_tree(tmp_path: Path) -> None:
    seq = tmp_path / "seq001"
    seq.mkdir()
    (seq / "calibration.json").write_text("{}", encoding="utf-8")
    (seq / "frame_0.0.pcd").write_text("DATA ascii\n0 0 1\n", encoding="utf-8")
    (seq / "left_0001.png").write_bytes(b"not-an-image-but-counted")
    (seq / "truth.csv").write_text("time_s,x_m,y_m,z_m\n0,0,0,1\n", encoding="utf-8")
    (seq / "recording.bag").write_bytes(b"bag-placeholder")

    summary = inspect_mmuad_layout(tmp_path)
    assert summary["category_counts"]["point_cloud"] == 1
    assert summary["category_counts"]["image"] == 1
    assert summary["category_counts"]["truth_or_label"] == 1
    assert summary["category_counts"]["rosbag_or_recording"] == 1
    assert summary["sequence_candidates"][0]["has_calibration"] is True
    assert any("ROS bag" in item for item in summary["recommendations"])


def test_binary_pcd_point_cloud_is_clustered(tmp_path: Path) -> None:
    import struct

    pcd = tmp_path / "frame_2.5.pcd"
    header = "\n".join(
        [
            "# .PCD v0.7",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            "WIDTH 3",
            "HEIGHT 1",
            "POINTS 3",
            "DATA binary",
            "",
        ]
    ).encode("ascii")
    payload = b"".join(
        struct.pack("<fff", x, y, z)
        for x, y, z in [(0.0, 0.0, 1.0), (0.1, 0.0, 1.1), (0.2, 0.1, 1.0)]
    )
    pcd.write_bytes(header + payload)
    frame = load_point_cloud_file_as_candidates(pcd, voxel_size_m=0.5, min_points=3)
    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 2.5) < 1e-9


def test_binary_compressed_pcd_point_cloud_is_clustered(tmp_path: Path) -> None:
    import struct

    pcd = tmp_path / "frame_3.5.pcd"
    header = "\n".join(
        [
            "# .PCD v0.7",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            "WIDTH 3",
            "HEIGHT 1",
            "POINTS 3",
            "DATA binary_compressed",
            "",
        ]
    ).encode("ascii")
    points = np.array(
        [(0.0, 0.0, 1.0), (0.1, 0.0, 1.1), (0.2, 0.1, 1.0)],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
    )
    uncompressed = points["x"].tobytes() + points["y"].tobytes() + points["z"].tobytes()
    compressed = _lzf_literal_payload(uncompressed)
    pcd.write_bytes(header + struct.pack("<II", len(compressed), len(uncompressed)) + compressed)

    frame = load_point_cloud_file_as_candidates(pcd, voxel_size_m=0.5, min_points=3)

    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 3.5) < 1e-9
    assert abs(float(frame.rows.loc[0, "x_m"]) - 0.1) < 1e-6


def test_binary_bin_point_cloud_is_clustered(tmp_path: Path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0, 0.5],
            [0.1, 0.0, 1.1, 0.6],
            [0.2, 0.1, 1.0, 0.7],
        ],
        dtype="<f4",
    )
    bin_path = tmp_path / "livox_points_4.25.bin"
    points.tofile(bin_path)

    frame = load_point_cloud_file_as_candidates(bin_path, voxel_size_m=0.5, min_points=3)

    assert len(frame.rows) == 1
    row = frame.rows.iloc[0]
    assert abs(float(row["time_s"]) - 4.25) < 1e-9
    assert abs(float(row["x_m"]) - 0.1) < 1e-6


def test_gzipped_ascii_ply_point_cloud_is_clustered(tmp_path: Path) -> None:
    ply = tmp_path / "frame_6.75.ply.gz"
    text = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            "element vertex 3",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
            "0 0 1",
            "0.1 0 1.1",
            "0.2 0.1 1.0",
        ]
    )
    with gzip.open(ply, "wt", encoding="utf-8") as handle:
        handle.write(text)

    frame = load_point_cloud_file_as_candidates(ply, voxel_size_m=0.5, min_points=3)

    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 6.75) < 1e-9


def test_binary_little_endian_ply_point_cloud_is_clustered(tmp_path: Path) -> None:
    import struct

    ply = tmp_path / "frame_8.5.ply"
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 3",
            "property float x",
            "property float y",
            "property float z",
            "property uchar intensity",
            "end_header",
            "",
        ]
    ).encode("ascii")
    payload = b"".join(
        struct.pack("<fffB", x, y, z, intensity)
        for x, y, z, intensity in [
            (0.0, 0.0, 1.0, 10),
            (0.1, 0.0, 1.1, 20),
            (0.2, 0.1, 1.0, 30),
        ]
    )
    ply.write_bytes(header + payload)

    frame = load_point_cloud_file_as_candidates(ply, voxel_size_m=0.5, min_points=3)

    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 8.5) < 1e-9
    assert abs(float(frame.rows.loc[0, "x_m"]) - 0.1) < 1e-6


def test_gzipped_binary_big_endian_ply_point_cloud_is_clustered(tmp_path: Path) -> None:
    import struct

    ply = tmp_path / "frame_9.25.ply.gz"
    header = "\n".join(
        [
            "ply",
            "format binary_big_endian 1.0",
            "element vertex 3",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
            "",
        ]
    ).encode("ascii")
    payload = b"".join(
        struct.pack(">fff", x, y, z)
        for x, y, z in [(0.0, 0.0, 1.0), (0.1, 0.0, 1.1), (0.2, 0.1, 1.0)]
    )
    with gzip.open(ply, "wb") as handle:
        handle.write(header + payload)

    frame = load_point_cloud_file_as_candidates(ply, voxel_size_m=0.5, min_points=3)

    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 9.25) < 1e-9
    assert abs(float(frame.rows.loc[0, "z_m"]) - (3.1 / 3.0)) < 1e-6


def test_las_point_cloud_is_clustered(tmp_path: Path) -> None:
    las = tmp_path / "frame_10.5.las"
    las.write_bytes(
        _minimal_las_bytes(
            [
                (0.0, 0.0, 1.0),
                (0.1, 0.0, 1.1),
                (0.2, 0.1, 1.0),
            ]
        )
    )

    frame = load_point_cloud_file_as_candidates(las, voxel_size_m=0.5, min_points=3)

    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 10.5) < 1e-9
    assert abs(float(frame.rows.loc[0, "x_m"]) - 0.1) < 1e-6


def test_laz_point_cloud_uses_optional_laspy_reader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    laz = tmp_path / "frame_10.5.laz"
    laz.write_bytes(b"not-real-laz-but-fake-laspy-reads-the-path")
    calls: list[Path] = []

    def fake_read(path: Path):
        calls.append(Path(path))
        return SimpleNamespace(
            x=np.array([0.0, 0.1, 0.2], dtype=float),
            y=np.array([0.0, 0.0, 0.1], dtype=float),
            z=np.array([1.0, 1.1, 1.0], dtype=float),
        )

    monkeypatch.setitem(sys.modules, "laspy", SimpleNamespace(read=fake_read))

    frame = load_point_cloud_file_as_candidates(laz, voxel_size_m=0.5, min_points=3)

    assert calls == [laz]
    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 10.5) < 1e-9
    assert abs(float(frame.rows.loc[0, "x_m"]) - 0.1) < 1e-6


def test_numpy_point_cloud_file_is_clustered(tmp_path: Path) -> None:
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.1], [0.2, 0.1, 1.0]])
    npy = tmp_path / "cloud_3.0.npy"
    np.save(npy, points)
    frame = load_point_cloud_file_as_candidates(npy, voxel_size_m=0.5, min_points=3)
    assert len(frame.rows) == 1
    assert abs(float(frame.rows.loc[0, "time_s"]) - 3.0) < 1e-9


def test_numpy_point_cloud_keeps_xyzt_column_order(tmp_path: Path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0, 10.0],
            [0.1, 0.0, 1.1, 10.0],
            [0.2, 0.1, 1.0, 10.0],
        ]
    )
    npy = tmp_path / "livox_points.npy"
    np.save(npy, points)

    frame = load_point_cloud_file_as_candidates(npy, voxel_size_m=0.5, min_points=3)

    assert len(frame.rows) == 1
    row = frame.rows.iloc[0]
    assert abs(float(row["time_s"]) - 10.0) < 1e-9
    assert abs(float(row["x_m"]) - 0.1) < 1e-9


def test_numpy_trajectory_files_use_time_xyz_column_order(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.npy"
    candidates_path = tmp_path / "trajectory.npz"
    trajectory = np.array(
        [
            [5.0, 100.0, 200.0, 30.0],
            [6.0, 101.0, 201.0, 31.0],
        ]
    )
    np.save(truth_path, trajectory)
    np.savez(candidates_path, trajectory=trajectory)

    truth = load_truth_file(truth_path, default_sequence_id="seq_numpy")
    candidates = load_candidate_file(
        candidates_path,
        default_sequence_id="seq_numpy",
        source="numpy-trajectory",
    )

    assert truth.rows["time_s"].tolist() == [5.0, 6.0]
    assert truth.rows["x_m"].tolist() == [100.0, 101.0]
    assert candidates.rows["source"].tolist() == ["numpy-trajectory", "numpy-trajectory"]
    assert candidates.rows["z_m"].tolist() == [30.0, 31.0]


def test_compact_numpy_trajectory_accepts_column_vector_frames(tmp_path: Path) -> None:
    truth_path = tmp_path / "12.5.npy"
    candidate_path = tmp_path / "track_13.5.npy"
    np.save(truth_path, np.array([[1.0], [2.0], [3.0]]))
    np.save(candidate_path, np.array([[4.0], [5.0], [6.0], [0.8]]))

    truth = load_truth_file(truth_path, default_sequence_id="seq_column")
    candidates = load_candidate_file(
        candidate_path,
        default_sequence_id="seq_column",
        source="column-vector-trajectory",
    )

    assert truth.rows["time_s"].tolist() == [12.5]
    assert truth.rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]
    assert candidates.rows["time_s"].tolist() == [13.5]
    assert candidates.rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [4.0, 5.0, 6.0]
    assert candidates.rows["confidence"].tolist() == [0.8]


def test_compact_text_trajectory_files_infer_timestamp_from_filename(
    tmp_path: Path,
) -> None:
    truth_path = tmp_path / "12.5.txt"
    candidate_path = tmp_path / "track_13.5.txt"
    truth_path.write_text("1.0 2.0 3.0\n", encoding="utf-8")
    candidate_path.write_text("4.0,5.0,6.0\n", encoding="utf-8")

    truth = load_truth_file(truth_path, default_sequence_id="seq_text")
    candidates = load_candidate_file(
        candidate_path,
        default_sequence_id="seq_text",
        source="text-trajectory",
    )

    assert truth.rows["time_s"].tolist() == [12.5]
    assert truth.rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]
    assert candidates.rows["time_s"].tolist() == [13.5]
    assert candidates.rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [4.0, 5.0, 6.0]
    assert candidates.rows["source"].tolist() == ["text-trajectory"]


def test_compact_text_trajectory_files_accept_time_xyz_rows(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.txt"
    path.write_text("0.0 1.0 2.0 3.0\n1.0 2.0 3.0 4.0\n", encoding="utf-8")

    candidates = load_candidate_file(
        path,
        default_sequence_id="seq_text_rows",
        source="text-trajectory",
    )

    assert candidates.rows["time_s"].tolist() == [0.0, 1.0]
    assert candidates.rows["x_m"].tolist() == [1.0, 2.0]
    assert candidates.rows["z_m"].tolist() == [3.0, 4.0]


def test_sequence_root_loads_compact_numpy_trajectory_and_truth_arrays(tmp_path: Path) -> None:
    seq = tmp_path / "seq_numpy"
    seq.mkdir()
    np.save(
        seq / "trajectory.npy",
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ]
        ),
    )
    np.save(
        seq / "truth.npy",
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ]
        ),
    )

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].candidate_trajectory_files == (seq / "trajectory.npy",)
    assert not discovered[0].point_cloud_files
    assert candidates.rows["time_s"].tolist() == [0.0, 1.0]
    assert truth is not None
    assert truth.rows["x_m"].tolist() == [0.0, 1.0]


def test_sequence_root_loads_mmuad_modality_folder_layout(tmp_path: Path) -> None:
    seq = tmp_path / "val" / "seq0001"
    livox = seq / "livox_avia"
    truth_dir = seq / "ground_truth"
    class_dir = seq / "class"
    livox.mkdir(parents=True)
    truth_dir.mkdir()
    class_dir.mkdir()
    timestamp = "1706255054.386069"
    np.save(
        livox / f"{timestamp}.npy",
        np.array(
            [
                [5.2, 17.4, 8.9],
                [5.3, 17.5, 9.0],
                [5.2, 17.5, 8.9],
            ]
        ),
    )
    np.save(truth_dir / f"{timestamp}.npy", np.array([5.25, 17.45, 8.95]))
    np.save(class_dir / f"{timestamp}.npy", np.array(2))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert [sequence.sequence_id for sequence in discovered] == ["seq0001"]
    assert discovered[0].point_cloud_files == (livox / f"{timestamp}.npy",)
    assert discovered[0].truth_file == truth_dir / f"{timestamp}.npy"
    assert discovered[0].truth_files == (truth_dir / f"{timestamp}.npy",)
    assert discovered[0].class_files == (class_dir / f"{timestamp}.npy",)
    assert len(candidates.rows) == 1
    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq0001"
    assert row["source"] == "livox_avia"
    assert row["class_name"] == "2"
    assert abs(float(row["time_s"]) - float(timestamp)) < 1.0e-9
    assert truth is not None
    truth_row = truth.rows.iloc[0]
    assert abs(float(truth_row["time_s"]) - float(timestamp)) < 1.0e-9
    assert abs(float(truth_row["x_m"]) - 5.25) < 1.0e-9
    output = run_mmuad_tracker(candidates, truth)
    assert output.estimates["class_name"].tolist() == ["2"]
    results = estimates_to_mmaud_results_frame(output.estimates, class_name="unknown")
    assert results["uav_type"].tolist() == ["2"]


def test_sequence_root_loads_official_track5_point_cloud_folders(tmp_path: Path) -> None:
    seq = tmp_path / "train" / "seq1"
    timestamp = "1706255054.386069"
    for folder, offset in (
        ("lidar_360", 0.0),
        ("livox_avia", 10.0),
        ("radar_enhance_pcl", 20.0),
    ):
        directory = seq / folder
        directory.mkdir(parents=True, exist_ok=True)
        np.save(
            directory / f"{timestamp}.npy",
            np.array(
                [
                    [1.0 + offset, 2.0, 3.0, 0.1],
                    [1.1 + offset, 2.0, 3.0, 0.2],
                    [1.0 + offset, 2.1, 3.0, 0.3],
                ]
            ),
        )
    truth_dir = seq / "ground_truth"
    truth_dir.mkdir()
    np.save(truth_dir / f"{timestamp}.npy", np.array([1.0, 2.0, 3.0]))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert [sequence.sequence_id for sequence in discovered] == ["seq1"]
    assert {
        path.relative_to(seq).as_posix()
        for path in discovered[0].point_cloud_files
    } == {
        f"lidar_360/{timestamp}.npy",
        f"livox_avia/{timestamp}.npy",
        f"radar_enhance_pcl/{timestamp}.npy",
    }
    assert candidates.rows["source"].tolist() == [
        "lidar_360",
        "livox_avia",
        "radar_enhance_pcl",
    ]
    assert candidates.rows["time_s"].tolist() == [float(timestamp)] * 3
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [float(timestamp)]


def test_official_track5_timestamp_template_prefers_truth_then_sensor_frames(
    tmp_path: Path,
) -> None:
    train_seq = tmp_path / "train" / "seq1"
    (train_seq / "ground_truth" / "leica").mkdir(parents=True)
    (train_seq / "Image").mkdir()
    (train_seq / "livox_avia").mkdir()
    np.save(train_seq / "ground_truth" / "leica" / "10.0.npy", np.array([0.0, 0.0, 0.0]))
    (train_seq / "Image" / "11.0.png").write_bytes(b"not-a-real-image")
    np.save(train_seq / "livox_avia" / "12.0.npy", np.zeros((3, 3)))

    val_seq = tmp_path / "val" / "seq2"
    (val_seq / "Image" / "front_camera").mkdir(parents=True)
    (val_seq / "livox_avia" / "stream0").mkdir(parents=True)
    (val_seq / "Image" / "front_camera" / "20.0.png").write_bytes(b"not-a-real-image")
    np.save(val_seq / "livox_avia" / "stream0" / "21.0.npy", np.zeros((3, 3)))

    discovered = {
        paths.sequence_id: paths for paths in discover_sequence_paths(tmp_path)
    }

    train_template = official_track5_timestamp_template(discovered["seq1"])
    val_template = official_track5_timestamp_template(discovered["seq2"])
    val_image_template = official_track5_timestamp_template(
        discovered["seq2"],
        timestamp_source="image",
    )

    assert train_template.rows["time_s"].tolist() == [10.0]
    assert val_template.rows["time_s"].tolist() == [20.0, 21.0]
    assert val_image_template.rows["time_s"].tolist() == [20.0]


def test_official_track5_timestamp_template_reads_timestamp_sidecars(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "val" / "seq_sidecar"
    image = seq / "Image" / "front_camera"
    livox = seq / "livox_avia"
    image.mkdir(parents=True)
    livox.mkdir()
    (image / "front_a.png").write_bytes(b"not-a-real-image")
    (image / "front_b.png").write_bytes(b"not-a-real-image")
    pd.DataFrame(
        {
            "frame": ["front_a.png", "front_b.png"],
            "timestamp_ns": [1_000_000_000, 2_250_000_000],
        }
    ).to_csv(image / "timestamps.csv", index=False)
    (livox / "timestamps.json").write_text(
        json.dumps({"timestamps": [3.0, 4.5]}),
        encoding="utf-8",
    )

    discovered = {
        paths.sequence_id: paths for paths in discover_sequence_paths(tmp_path)
    }

    image_template = official_track5_timestamp_template(
        discovered["seq_sidecar"],
        timestamp_source="image",
    )
    all_template = official_track5_timestamp_template(discovered["seq_sidecar"])

    assert image_template.rows["time_s"].tolist() == [1.0, 2.25]
    assert all_template.rows["time_s"].tolist() == [1.0, 2.25, 3.0, 4.5]


def test_official_track5_timestamp_template_reads_numpy_timestamp_sidecars(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "val" / "seq_numpy_sidecar"
    image = seq / "Image"
    livox = seq / "livox_avia"
    image.mkdir(parents=True)
    livox.mkdir()
    np.save(image / "timestamps.npy", np.array([7.0, 8.25]))
    np.savez(livox / "frame_times.npz", frame_times=np.array([[0.0, 9.0], [1.0, 10.5]]))

    discovered = {
        paths.sequence_id: paths for paths in discover_sequence_paths(tmp_path)
    }

    image_template = official_track5_timestamp_template(
        discovered["seq_numpy_sidecar"],
        timestamp_source="image",
    )
    all_template = official_track5_timestamp_template(discovered["seq_numpy_sidecar"])

    assert image_template.rows["time_s"].tolist() == [7.0, 8.25]
    assert all_template.rows["time_s"].tolist() == [7.0, 8.25, 9.0, 10.5]


def test_official_track5_timestamp_template_reads_text_frame_lists(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "val" / "seq_text_frames"
    image = seq / "Image"
    image.mkdir(parents=True)
    (image / "front_a.png").write_bytes(b"not-a-real-image")
    (image / "front_b.png").write_bytes(b"not-a-real-image")
    (image / "frame_times.txt").write_text(
        "front_a.png 5.0\nfront_b.png 6.5\n",
        encoding="utf-8",
    )

    discovered = {
        paths.sequence_id: paths for paths in discover_sequence_paths(tmp_path)
    }
    image_template = official_track5_timestamp_template(
        discovered["seq_text_frames"],
        timestamp_source="image",
    )

    assert image_template.rows["time_s"].tolist() == [5.0, 6.5]


def test_cli_completes_official_results_to_sequence_timestamps(tmp_path: Path) -> None:
    root = tmp_path / "data"
    seq = root / "val" / "seq1"
    image = seq / "Image"
    livox = seq / "livox_avia"
    image.mkdir(parents=True)
    livox.mkdir()
    for timestamp in ("0.0", "1.0"):
        (image / f"{timestamp}.png").write_bytes(b"not-a-real-image")
    for timestamp, x in (("0.0", 0.0), ("2.0", 2.0)):
        np.save(
            livox / f"{timestamp}.npy",
            np.array(
                [
                    [x, 0.0, 10.0],
                    [x + 0.1, 0.0, 10.0],
                    [x, 0.1, 10.0],
                ]
            ),
        )
    output = tmp_path / "out"
    results = output / "mmaud_results.csv"
    zip_path = output / "submission.zip"

    status = mmuad_cli_main(
        [
            "--sequence-root",
            str(root),
            "--split-name",
            "val",
            "--voxel-size-m",
            "0.5",
            "--min-cluster-points",
            "3",
            "--completion-max-interpolation-gap-s",
            "3.0",
            "--ug2-official-complete-to-sequence-timestamps",
            "--ug2-official-timestamp-source",
            "image",
            "--ug2-official-results-csv",
            str(results),
            "--ug2-official-codabench-zip",
            str(zip_path),
            "--ug2-official-validate-on-write",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    rows = pd.read_csv(results)
    assert rows.columns.tolist() == [
        "Sequence",
        "Timestamp",
        "Position",
        "Classification",
    ]
    assert rows["Sequence"].tolist() == ["seq1", "seq1"]
    assert rows["Timestamp"].tolist() == [0.0, 1.0]
    assert (output / "mmuad_official_timestamp_completion_rows.csv").exists()
    summary = json.loads(
        (output / "mmuad_official_timestamp_completion_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["requested_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["timestamp_source"] == "image"
    assert summary["sequences"]["seq1"]["completed_count"] == 2
    assert summary["sequences"]["seq1"]["all_requested_timestamps_completed"] is True
    validation = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    validation_rows = pd.read_csv(output / "mmuad_official_submission_validation_rows.csv")
    assert validation["valid"] is True
    assert validation["template_checked"] is True
    assert validation["template_timestamp_count"] == 2
    assert validation["missing_template_timestamp_count"] == 0
    assert set(validation_rows["status"]) == {"ok", "covered_template_timestamp"}
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
        zipped = pd.read_csv(archive.open("mmaud_results.csv"))
    assert zipped["Timestamp"].tolist() == [0.0, 1.0]


def test_cli_sequence_root_runs_normalized_zip_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "mmuad_normalized.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "seq_archive/candidates.csv",
            "\n".join(
                [
                    "sequence_id,time_s,source,x_m,y_m,z_m",
                    "seq_archive,0.0,radar,0.0,0.0,10.0",
                    "seq_archive,1.0,radar,1.0,0.0,10.0",
                ]
            ),
        )
        archive.writestr(
            "seq_archive/truth.csv",
            "\n".join(
                [
                    "sequence_id,time_s,x_m,y_m,z_m",
                    "seq_archive,0.0,0.0,0.0,10.0",
                    "seq_archive,1.0,1.0,0.0,10.0",
                ]
            ),
        )
    output = tmp_path / "zip_out"

    status = mmuad_cli_main(
        [
            "--sequence-root",
            str(archive_path),
            "--output-dir",
            str(output),
            "--submission-csv",
            str(output / "submission.csv"),
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output / "mmuad_estimates.csv")
    assert estimates["sequence_id"].tolist() == ["seq_archive", "seq_archive"]
    assert estimates["time_s"].tolist() == [0.0, 1.0]
    metrics = json.loads((output / "mmuad_metrics.json").read_text(encoding="utf-8"))
    assert metrics["pooled"]["mean_3d_m"] < 1.0
    archive_manifest = json.loads(
        (output / "mmuad_sequence_root_archive_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert archive_manifest["schema"] == "raft-uav-mmuad-archive-extraction-v1"
    assert archive_manifest["archive_format"] == "zip"
    assert archive_manifest["extracted_file_count"] == 2
    assert archive_manifest["skipped_member_count"] == 0
    assert Path(archive_manifest["extract_root"]).is_dir()


def test_mmuad_archive_extraction_skips_unsafe_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "escape")
        archive.writestr("seq_safe/candidates.csv", "time_s,x_m,y_m,z_m\n0,0,0,1\n")

    manifest = extract_mmuad_archive(archive_path, tmp_path / "extract")

    assert manifest["extracted_file_count"] == 1
    assert manifest["skipped_member_count"] == 1
    assert manifest["skipped_members"] == [
        {"member": "../outside.txt", "reason": "unsafe_member_path"}
    ]
    assert not (tmp_path / "outside.txt").exists()
    assert (Path(manifest["extract_root"]) / "seq_safe" / "candidates.csv").exists()


def test_cli_completes_official_results_from_zipped_track5_frames(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "mmuad_track5.zip"

    def npy_payload(points: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, points)
        return buffer.getvalue()

    with ZipFile(archive_path, "w") as archive:
        for timestamp in ("0.0", "1.0"):
            archive.writestr(
                f"val/seq_zip/Image/{timestamp}.png",
                b"not-a-real-image",
            )
        for timestamp, x in (("0.0", 0.0), ("2.0", 2.0)):
            archive.writestr(
                f"val/seq_zip/livox_avia/{timestamp}.npy",
                npy_payload(
                    np.array(
                        [
                            [x, 0.0, 10.0],
                            [x + 0.1, 0.0, 10.0],
                            [x, 0.1, 10.0],
                        ]
                    )
                ),
            )
    output = tmp_path / "track5_zip_out"
    results = output / "mmaud_results.csv"
    zip_path = output / "official_submission.zip"

    status = mmuad_cli_main(
        [
            "--sequence-root",
            str(archive_path),
            "--split-name",
            "val",
            "--voxel-size-m",
            "0.5",
            "--min-cluster-points",
            "3",
            "--completion-max-interpolation-gap-s",
            "3.0",
            "--ug2-official-complete-to-sequence-timestamps",
            "--ug2-official-timestamp-source",
            "image",
            "--ug2-official-results-csv",
            str(results),
            "--ug2-official-codabench-zip",
            str(zip_path),
            "--ug2-official-classification",
            "2",
            "--ug2-official-validate-on-write",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    rows = pd.read_csv(results)
    assert rows["Sequence"].tolist() == ["seq_zip", "seq_zip"]
    assert rows["Timestamp"].tolist() == [0.0, 1.0]
    assert rows["Classification"].tolist() == [2, 2]
    validation = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["codabench_upload_ready"] is True
    assert validation["template_timestamp_count"] == 2
    archive_manifest = json.loads(
        (output / "mmuad_sequence_root_archive_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert archive_manifest["archive_format"] == "zip"
    assert archive_manifest["extracted_file_count"] == 4
    assert all(".." not in row["member"] for row in archive_manifest["extracted_files"])


def test_cli_completes_official_results_to_template_file_without_sequence_root(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates.csv"
    template = tmp_path / "official_template.csv"
    output = tmp_path / "out"
    results = output / "official_mmaud_results.csv"
    zip_path = output / "official_submission.zip"
    pd.DataFrame(
        {
            "sequence_id": ["seq_template", "seq_template"],
            "time_s": [0.0, 2.0],
            "source": ["radar", "radar"],
            "x_m": [0.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
        }
    ).to_csv(candidates, index=False)
    pd.DataFrame(
        {
            "Sequence": ["seq_template", "seq_template"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2],
        }
    ).to_csv(template, index=False)

    status = mmuad_cli_main(
        [
            "--candidate-csv",
            str(candidates),
            "--completion-max-interpolation-gap-s",
            "3.0",
            "--official-validation-template-file",
            str(template),
            "--ug2-official-complete-to-sequence-timestamps",
            "--ug2-official-results-csv",
            str(results),
            "--ug2-official-codabench-zip",
            str(zip_path),
            "--ug2-official-classification",
            "2",
            "--ug2-official-validate-on-write",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    rows = pd.read_csv(results)
    assert rows["Sequence"].tolist() == ["seq_template", "seq_template"]
    assert rows["Timestamp"].tolist() == [0.0, 1.0]
    assert rows["Classification"].tolist() == [2, 2]
    summary = json.loads(
        (output / "mmuad_official_timestamp_completion_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["requested_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["sequences"]["seq_template"]["completed_count"] == 2
    assert summary["sequences"]["seq_template"]["completion_coverage_fraction"] == 1.0
    validation = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["valid"] is True
    assert validation["template_checked"] is True
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
        zipped = pd.read_csv(archive.open("mmaud_results.csv"))
    assert zipped["Timestamp"].tolist() == [0.0, 1.0]


def test_cli_validate_on_write_rejects_incomplete_official_zip(tmp_path: Path) -> None:
    root = tmp_path / "data"
    seq = root / "val" / "seq1"
    image = seq / "Image"
    image.mkdir(parents=True)
    for timestamp in ("0.0", "1.0"):
        (image / f"{timestamp}.png").write_bytes(b"not-a-real-image")
    pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [0.0],
            "source": ["radar"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [10.0],
        }
    ).to_csv(seq / "candidates.csv", index=False)
    class_map = tmp_path / "class_map.csv"
    class_map.write_text("sequence_id,uav_type\nseq1,2\n", encoding="utf-8")
    output = tmp_path / "out"

    with pytest.raises(SystemExit, match="failed validation"):
        mmuad_cli_main(
            [
                "--sequence-root",
                str(root),
                "--split-name",
                "val",
                "--ug2-official-timestamp-source",
                "image",
                "--ug2-class-map-file",
                str(class_map),
                "--ug2-official-codabench-zip",
                str(output / "submission.zip"),
                "--ug2-official-validate-on-write",
                "--output-dir",
                str(output),
            ]
        )

    summary = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    rows = pd.read_csv(output / "mmuad_official_submission_validation_rows.csv")
    assert summary["valid"] is False
    assert summary["template_checked"] is True
    assert summary["missing_template_timestamp_count"] == 1
    assert "missing_template_timestamp" in rows["status"].tolist()
    manifest = json.loads(
        (output / "mmuad_official_upload_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["valid"] is False
    assert manifest["codabench_upload_ready"] is False
    assert manifest["leaderboard_ready"] is False
    assert manifest["blocking_sequences"] == ["seq1"]
    assert manifest["sequences"]["seq1"]["missing_template_timestamp_count"] == 1


def test_cli_validates_official_zip_against_sequence_timestamps(tmp_path: Path) -> None:
    root = tmp_path / "data"
    seq = root / "val" / "seq1"
    image = seq / "Image"
    image.mkdir(parents=True)
    for timestamp in ("0.0", "1.0"):
        (image / f"{timestamp}.png").write_bytes(b"not-a-real-image")
    zip_path = tmp_path / "official_submission.zip"
    manifest_path = tmp_path / "official_upload_manifest.json"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(1,0,10)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", official.to_csv(index=False))
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--validate-ug2-official-codabench-zip",
            str(zip_path),
            "--sequence-root",
            str(root),
            "--split-name",
            "val",
            "--ug2-official-timestamp-source",
            "image",
            "--official-validation-json",
            str(output / "validation.json"),
            "--official-validation-rows-csv",
            str(output / "validation_rows.csv"),
            "--official-upload-manifest-json",
            str(manifest_path),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    summary = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(output / "validation_rows.csv")
    assert summary["valid"] is True
    assert summary["template_checked"] is True
    assert summary["template_timestamp_count"] == 2
    assert summary["missing_template_timestamp_count"] == 0
    assert summary["extra_prediction_count"] == 0
    assert set(rows["status"]) == {"ok", "covered_template_timestamp"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_path"] == str(zip_path)
    assert manifest["validation_json"] == str(output / "validation.json")
    assert manifest["validation_rows_csv"] == str(output / "validation_rows.csv")
    assert manifest["artifact_exists"] is True
    assert manifest["artifact_size_bytes"] == summary["artifact_size_bytes"]
    assert manifest["artifact_sha256"] == summary["artifact_sha256"]
    assert manifest["mmaud_results_csv_size_bytes"] == summary["mmaud_results_csv_size_bytes"]
    assert manifest["mmaud_results_csv_compressed_size_bytes"] == summary[
        "mmaud_results_csv_compressed_size_bytes"
    ]
    assert manifest["mmaud_results_csv_crc32"] == summary["mmaud_results_csv_crc32"]
    assert manifest["mmaud_results_csv_sha256"] == summary["mmaud_results_csv_sha256"]
    assert manifest["codabench_upload_ready"] is True
    assert manifest["leaderboard_ready"] is True
    assert manifest["sequence_count"] == 1
    assert manifest["ready_sequence_count"] == 1
    assert manifest["sequences"]["seq1"]["covered_template_timestamp_count"] == 2
    verification_path = output / "manifest_verification.json"
    verify_status = mmuad_cli_main(
        [
            "--verify-official-upload-manifest",
            str(manifest_path),
            "--official-upload-manifest-verification-json",
            str(verification_path),
            "--output-dir",
            str(output),
        ]
    )
    assert verify_status == 0
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["valid"] is True
    assert verification["codabench_upload_ready"] is True
    assert verification["artifact_size_matches"] is True
    assert verification["artifact_sha256_matches"] is True
    assert verification["mmaud_results_csv_present"] is True
    assert verification["mmaud_results_csv_size_matches"] is True
    assert verification["mmaud_results_csv_compressed_size_matches"] is True
    assert verification["mmaud_results_csv_crc32_matches"] is True
    assert verification["mmaud_results_csv_sha256_matches"] is True
    assert verification["validation_json_exists"] is True
    assert verification["validation_rows_csv_exists"] is True


def test_cli_official_upload_manifest_verification_rejects_tampered_zip(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    seq = root / "val" / "seq1"
    image = seq / "Image"
    image.mkdir(parents=True)
    for timestamp in ("0.0", "1.0"):
        (image / f"{timestamp}.png").write_bytes(b"not-a-real-image")
    zip_path = tmp_path / "official_submission.zip"
    manifest_path = tmp_path / "official_upload_manifest.json"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(1,0,10)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", official.to_csv(index=False))
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--validate-ug2-official-codabench-zip",
            str(zip_path),
            "--sequence-root",
            str(root),
            "--split-name",
            "val",
            "--ug2-official-timestamp-source",
            "image",
            "--official-upload-manifest-json",
            str(manifest_path),
            "--output-dir",
            str(output),
        ]
    )
    assert status == 0

    tampered = official.copy()
    tampered.loc[1, "Position"] = "(999,0,10)"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", tampered.to_csv(index=False))

    verification_path = output / "tampered_manifest_verification.json"
    verify_status = mmuad_cli_main(
        [
            "--verify-official-upload-manifest",
            str(manifest_path),
            "--official-upload-manifest-verification-json",
            str(verification_path),
            "--output-dir",
            str(output),
        ]
    )

    assert verify_status == 1
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["valid"] is False
    assert verification["codabench_upload_ready"] is False
    assert verification["artifact_sha256_matches"] is False
    assert verification["mmaud_results_csv_sha256_matches"] is False
    assert any("sha256 mismatch" in error for error in verification["errors"])


def test_cli_official_zip_validation_requires_template_for_upload_ready(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "official_submission.zip"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,10)"],
            "Classification": [2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", official.to_csv(index=False))
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--validate-ug2-official-codabench-zip",
            str(zip_path),
            "--official-validation-json",
            str(output / "validation.json"),
            "--official-validation-rows-csv",
            str(output / "validation_rows.csv"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 1
    summary = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(output / "validation_rows.csv")
    assert summary["valid"] is True
    assert summary["template_checked"] is False
    assert summary["leaderboard_ready"] is False
    assert summary["codabench_upload_ready"] is False
    assert summary["leaderboard_blocking_reasons"] == [
        "timestamp_template_not_checked"
    ]
    assert rows["status"].tolist() == ["ok"]


def test_cli_official_zip_validation_rejects_empty_template_for_upload_ready(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "official_submission.zip"
    template = tmp_path / "empty_template.csv"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,10)"],
            "Classification": [2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", official.to_csv(index=False))
    pd.DataFrame(columns=["sequence_id", "time_s"]).to_csv(template, index=False)
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--validate-ug2-official-codabench-zip",
            str(zip_path),
            "--official-validation-template-file",
            str(template),
            "--official-validation-json",
            str(output / "validation.json"),
            "--official-validation-rows-csv",
            str(output / "validation_rows.csv"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 1
    summary = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(output / "validation_rows.csv")
    assert summary["valid"] is True
    assert summary["template_checked"] is True
    assert summary["template_timestamp_count"] == 0
    assert summary["leaderboard_ready"] is False
    assert summary["codabench_upload_ready"] is False
    assert summary["leaderboard_blocking_reasons"] == ["no_template_timestamps"]
    assert rows["status"].tolist() == ["ok"]


def test_official_track5_validation_rejects_nested_results_member(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "nested_official_submission.zip"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,10)"],
            "Classification": [2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("submission/mmaud_results.csv", official.to_csv(index=False))

    validation = validate_official_track5_submission(zip_path)
    summary = validation.summary

    assert summary["valid"] is False
    assert summary["has_mmaud_results_csv"] is False
    assert summary["has_root_mmaud_results_csv"] is False
    assert summary["contains_only_mmaud_results_csv"] is False
    assert summary["nested_mmaud_results_csv_members"] == [
        "submission/mmaud_results.csv"
    ]
    assert "submission/mmaud_results.csv" in summary["file_members"]
    assert any("archive root" in error for error in summary["errors"])
    assert validation.rows.empty


def test_cli_normalizes_nested_official_submission_zip(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "nested_official_submission.zip"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [1.0, 0.0],
            "Position": ["array([1, 0, 10])", "(0,0,10)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("submission/mmaud_results.csv", official.to_csv(index=False))
        archive.writestr("README.txt", "not part of the upload")
    template = tmp_path / "template.csv"
    pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(1,0,10)"],
            "Classification": [2, 2],
        }
    ).to_csv(template, index=False)
    output = tmp_path / "out"
    normalized_zip = output / "normalized_submission.zip"
    normalized_csv = output / "mmaud_results.csv"

    status = mmuad_cli_main(
        [
            "--normalize-ug2-official-submission",
            str(source_zip),
            "--normalized-ug2-official-codabench-zip",
            str(normalized_zip),
            "--normalized-ug2-official-results-csv",
            str(normalized_csv),
            "--official-normalization-json",
            str(output / "normalization.json"),
            "--official-validation-template-file",
            str(template),
            "--official-validation-json",
            str(output / "validation.json"),
            "--official-validation-rows-csv",
            str(output / "validation_rows.csv"),
            "--official-upload-manifest-json",
            str(output / "upload_manifest.json"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    with ZipFile(normalized_zip) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
        normalized = pd.read_csv(archive.open("mmaud_results.csv"))
    assert normalized["Timestamp"].tolist() == [0.0, 1.0]
    assert normalized["Position"].tolist() == ["(0,0,10)", "(1,0,10)"]
    normalization = json.loads((output / "normalization.json").read_text(encoding="utf-8"))
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "upload_manifest.json").read_text(encoding="utf-8"))
    assert normalization["source_selection"] == "nested_mmaud_results_csv"
    assert normalization["source_member"] == "submission/mmaud_results.csv"
    assert validation["codabench_upload_ready"] is True
    assert manifest["codabench_upload_ready"] is True
    assert normalized_csv.exists()
    assert (output / "validation_rows.csv").exists()


def test_cli_validates_official_zip_against_official_template_zip(tmp_path: Path) -> None:
    submission_zip = tmp_path / "official_submission.zip"
    submission = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,10)"],
            "Classification": [2],
        }
    )
    with ZipFile(submission_zip, "w") as archive:
        archive.writestr("mmaud_results.csv", submission.to_csv(index=False))
    template_zip = tmp_path / "official_template.zip"
    template = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(template_zip, "w") as archive:
        archive.writestr("mmaud_results.csv", template.to_csv(index=False))
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--validate-ug2-official-codabench-zip",
            str(submission_zip),
            "--official-validation-template-file",
            str(template_zip),
            "--official-validation-json",
            str(output / "validation.json"),
            "--official-validation-rows-csv",
            str(output / "validation_rows.csv"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 1
    summary = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(output / "validation_rows.csv")
    assert summary["template_checked"] is True
    assert summary["template_timestamp_count"] == 2
    assert summary["missing_template_timestamp_count"] == 1
    assert "missing_template_timestamp" in rows["status"].tolist()


def test_sequence_root_loads_binary_livox_point_cloud_export(tmp_path: Path) -> None:
    seq = tmp_path / "seq_livox_bin"
    livox = seq / "livox_avia"
    truth_dir = seq / "ground_truth"
    livox.mkdir(parents=True)
    truth_dir.mkdir()
    timestamp = "12.75"
    np.array(
        [
            [3.0, 4.0, 5.0, 0.5],
            [3.1, 4.0, 5.1, 0.6],
            [3.0, 4.1, 5.0, 0.7],
        ],
        dtype="<f4",
    ).tofile(livox / f"{timestamp}.bin")
    np.save(truth_dir / f"{timestamp}.npy", np.array([3.0, 4.0, 5.0]))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert [sequence.sequence_id for sequence in discovered] == ["seq_livox_bin"]
    assert discovered[0].point_cloud_files == (livox / f"{timestamp}.bin",)
    assert len(candidates.rows) == 1
    row = candidates.rows.iloc[0]
    assert row["source"] == "livox_avia"
    assert abs(float(row["time_s"]) - 12.75) < 1e-9
    assert truth is not None


def test_sequence_root_uses_point_cloud_timestamp_sidecar_by_filename(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_livox_sidecar"
    livox = seq / "livox_avia"
    livox.mkdir(parents=True)
    bin_path = livox / "frame_a.bin"
    np.array(
        [
            [3.0, 4.0, 5.0, 0.5],
            [3.1, 4.0, 5.1, 0.6],
            [3.0, 4.1, 5.0, 0.7],
        ],
        dtype="<f4",
    ).tofile(bin_path)
    pd.DataFrame(
        {
            "filename": ["frame_a.bin"],
            "time_s": [12.75],
        }
    ).to_csv(livox / "timestamps.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert discovered[0].point_cloud_files == (bin_path,)
    assert len(candidates.rows) == 1
    assert abs(float(candidates.rows.loc[0, "time_s"]) - 12.75) < 1e-9
    assert truth is None


def test_sequence_root_uses_ordered_point_cloud_timestamp_sidecar(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_livox_ordered_sidecar"
    livox = seq / "livox_avia"
    livox.mkdir(parents=True)
    for frame_name, x_offset in (("frame_a.bin", 0.0), ("frame_b.bin", 10.0)):
        np.array(
            [
                [3.0 + x_offset, 4.0, 5.0, 0.5],
                [3.1 + x_offset, 4.0, 5.1, 0.6],
                [3.0 + x_offset, 4.1, 5.0, 0.7],
            ],
            dtype="<f4",
        ).tofile(livox / frame_name)
    (livox / "frame_times.txt").write_text("10.0\n11.5\n", encoding="utf-8")

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert discovered[0].point_cloud_files == (
        livox / "frame_a.bin",
        livox / "frame_b.bin",
    )
    assert candidates.rows["time_s"].tolist() == [10.0, 11.5]
    assert truth is None


def test_sequence_root_loads_gzipped_binary_livox_point_cloud_export(tmp_path: Path) -> None:
    seq = tmp_path / "seq_livox_bin_gz"
    livox = seq / "livox_avia"
    truth_dir = seq / "ground_truth"
    livox.mkdir(parents=True)
    truth_dir.mkdir()
    timestamp = "12.75"
    points = np.array(
        [
            [3.0, 4.0, 5.0, 0.5],
            [3.1, 4.0, 5.1, 0.6],
            [3.0, 4.1, 5.0, 0.7],
        ],
        dtype="<f4",
    )
    bin_path = livox / f"{timestamp}.bin.gz"
    with gzip.open(bin_path, "wb") as handle:
        handle.write(points.tobytes())
    np.save(truth_dir / f"{timestamp}.npy", np.array([3.0, 4.0, 5.0]))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert [sequence.sequence_id for sequence in discovered] == ["seq_livox_bin_gz"]
    assert discovered[0].point_cloud_files == (bin_path,)
    assert len(candidates.rows) == 1
    row = candidates.rows.iloc[0]
    assert row["source"] == "livox_avia"
    assert abs(float(row["time_s"]) - 12.75) < 1e-9
    assert truth is not None


def test_sequence_root_loads_gzipped_las_point_cloud_export(tmp_path: Path) -> None:
    seq = tmp_path / "seq_livox_las_gz"
    livox = seq / "livox_avia"
    truth_dir = seq / "ground_truth"
    livox.mkdir(parents=True)
    truth_dir.mkdir()
    timestamp = "12.75"
    las_path = livox / f"{timestamp}.las.gz"
    with gzip.open(las_path, "wb") as handle:
        handle.write(
            _minimal_las_bytes(
                [
                    (3.0, 4.0, 5.0),
                    (3.1, 4.0, 5.1),
                    (3.0, 4.1, 5.0),
                ]
            )
        )
    np.save(truth_dir / f"{timestamp}.npy", np.array([3.0, 4.0, 5.0]))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert [sequence.sequence_id for sequence in discovered] == ["seq_livox_las_gz"]
    assert discovered[0].point_cloud_files == (las_path,)
    assert len(candidates.rows) == 1
    row = candidates.rows.iloc[0]
    assert row["source"] == "livox_avia"
    assert abs(float(row["time_s"]) - 12.75) < 1e-9
    assert truth is not None


def test_sequence_root_loads_laz_point_cloud_export(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seq = tmp_path / "seq_laz"
    livox = seq / "livox_avia"
    livox.mkdir(parents=True)
    laz = livox / "12.75.laz"
    laz.write_bytes(b"not-real-laz-but-fake-laspy-reads-the-path")

    def fake_read(path: Path):
        assert Path(path) == laz
        return SimpleNamespace(
            x=np.array([3.0, 3.1, 3.0], dtype=float),
            y=np.array([4.0, 4.0, 4.1], dtype=float),
            z=np.array([5.0, 5.1, 5.0], dtype=float),
        )

    monkeypatch.setitem(sys.modules, "laspy", SimpleNamespace(read=fake_read))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert [sequence.sequence_id for sequence in discovered] == ["seq_laz"]
    assert discovered[0].point_cloud_files == (laz,)
    assert truth is None
    assert len(candidates.rows) == 1
    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_laz"
    assert row["source"] == "livox_avia"
    assert abs(float(row["time_s"]) - 12.75) < 1.0e-9
    assert abs(float(row["x_m"]) - (3.0 + 3.1 + 3.0) / 3.0) < 1.0e-9


def test_sequence_root_loads_json_livox_point_cloud_export(tmp_path: Path) -> None:
    seq = tmp_path / "seq_livox_json"
    livox = seq / "livox_avia"
    truth_dir = seq / "ground_truth"
    livox.mkdir(parents=True)
    truth_dir.mkdir()
    timestamp = "12.75"
    (livox / f"{timestamp}.json").write_text(
        json.dumps(
            {
                "points": [
                    {"x": 3.0, "y": 4.0, "z": 5.0},
                    {"x": 3.1, "y": 4.0, "z": 5.1},
                    {"x": 3.0, "y": 4.1, "z": 5.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    np.save(truth_dir / f"{timestamp}.npy", np.array([3.0, 4.0, 5.0]))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert [sequence.sequence_id for sequence in discovered] == ["seq_livox_json"]
    assert discovered[0].point_cloud_files == (livox / f"{timestamp}.json",)
    assert len(candidates.rows) == 1
    row = candidates.rows.iloc[0]
    assert row["source"] == "livox_avia"
    assert abs(float(row["time_s"]) - 12.75) < 1e-9
    assert truth is not None


def test_sequence_root_loads_nested_tracking_results_as_candidates(tmp_path: Path) -> None:
    seq = tmp_path / "seq_result"
    results = seq / "tracking_results"
    truth_dir = seq / "ground_truth"
    results.mkdir(parents=True)
    truth_dir.mkdir()
    np.save(results / "20.0.npy", np.array([1.0, 2.0, 3.0]))
    np.save(results / "20.1.npy", np.array([1.1, 2.1, 3.1]))
    np.save(truth_dir / "20.0.npy", np.array([1.0, 2.0, 3.0]))
    np.save(truth_dir / "20.1.npy", np.array([1.1, 2.1, 3.1]))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].candidate_trajectory_files == (
        results / "20.0.npy",
        results / "20.1.npy",
    )
    assert candidates.rows["source"].tolist() == ["tracking_results", "tracking_results"]
    assert candidates.rows["time_s"].tolist() == [20.0, 20.1]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [20.0, 20.1]


def test_sequence_root_loads_column_vector_numpy_frames_from_official_folders(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_column_vector"
    results = seq / "tracking_results"
    truth_dir = seq / "ground_truth"
    results.mkdir(parents=True)
    truth_dir.mkdir()
    np.save(results / "20.5.npy", np.array([[1.0], [2.0], [3.0]]))
    np.save(truth_dir / "20.5.npy", np.array([[1.1], [2.1], [3.1]]))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].candidate_trajectory_files == (results / "20.5.npy",)
    assert discovered[0].truth_file == truth_dir / "20.5.npy"
    assert candidates.rows["source"].tolist() == ["tracking_results"]
    assert candidates.rows["time_s"].tolist() == [20.5]
    assert candidates.rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [20.5]
    assert truth.rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.1, 2.1, 3.1]


def test_sequence_root_loads_compact_text_frames_from_official_folders(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_text_frames"
    results = seq / "tracking_results"
    truth_dir = seq / "ground_truth"
    results.mkdir(parents=True)
    truth_dir.mkdir()
    (results / "20.5.txt").write_text("1.0 2.0 3.0\n", encoding="utf-8")
    (truth_dir / "20.5.txt").write_text("1.1 2.1 3.1\n", encoding="utf-8")

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].candidate_csvs == (results / "20.5.txt",)
    assert discovered[0].truth_file == truth_dir / "20.5.txt"
    assert candidates.rows["source"].tolist() == ["tracking_results"]
    assert candidates.rows["time_s"].tolist() == [20.5]
    assert candidates.rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [20.5]
    assert truth.rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.1, 2.1, 3.1]


def test_sequence_root_preserves_nested_candidate_csv_source_hint(tmp_path: Path) -> None:
    seq = tmp_path / "seq_csv_result"
    results = seq / "tracking_results"
    truth_dir = seq / "ground_truth"
    results.mkdir(parents=True)
    truth_dir.mkdir()
    pd.DataFrame(
        {
            "time_s": [20.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    ).to_csv(results / "detections.csv", index=False)
    pd.DataFrame(
        {
            "time_s": [20.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    ).to_csv(truth_dir / "truth.csv", index=False)

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(discovered[0])

    assert discovered[0].candidate_csvs == (results / "detections.csv",)
    assert candidates.rows["source"].tolist() == ["tracking_results"]
    assert truth is not None


def test_sequence_glob_matches_sequences_inside_split_folders(tmp_path: Path) -> None:
    keep = tmp_path / "val" / "seq_keep"
    skip = tmp_path / "val" / "ignore_me"
    for seq in (keep, skip):
        seq.mkdir(parents=True)
        pd.DataFrame(
            {
                "time_s": [0.0],
                "x_m": [0.0],
                "y_m": [0.0],
                "z_m": [1.0],
            }
        ).to_csv(seq / "candidates.csv", index=False)

    discovered = discover_sequence_paths(tmp_path, sequence_glob="seq_*")

    assert [sequence.sequence_id for sequence in discovered] == ["seq_keep"]


def test_sequence_root_recurses_through_nested_grouping_folders(
    tmp_path: Path,
) -> None:
    keep = tmp_path / "val" / "fog" / "seq_keep"
    train = tmp_path / "train" / "clear" / "seq_train"
    skip = tmp_path / "val" / "fog" / "ignore_me"
    for seq in (keep, train, skip):
        seq.mkdir(parents=True)
        pd.DataFrame(
            {
                "time_s": [0.0],
                "x_m": [1.0],
                "y_m": [2.0],
                "z_m": [3.0],
            }
        ).to_csv(seq / "candidates.csv", index=False)

    discovered = discover_sequence_paths(tmp_path, sequence_glob="seq_*")
    val_discovered = discover_sequence_paths(tmp_path, sequence_glob="val/*/seq_*")

    assert [sequence.root.relative_to(tmp_path).as_posix() for sequence in discovered] == [
        "train/clear/seq_train",
        "val/fog/seq_keep",
    ]
    assert [sequence.root.relative_to(tmp_path).as_posix() for sequence in val_discovered] == [
        "val/fog/seq_keep",
    ]


def test_sequence_root_loads_wrapped_modality_folders(tmp_path: Path) -> None:
    seq = tmp_path / "val" / "fog" / "seq_wrapped"
    livox = seq / "sensors" / "livox_avia" / "stream0"
    results = seq / "outputs" / "tracking_results" / "fused"
    truth_dir = seq / "labels" / "ground_truth" / "leica"
    livox.mkdir(parents=True)
    results.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    timestamp = "20.0"
    np.save(
        livox / f"{timestamp}.npy",
        np.array(
            [
                [1.0, 2.0, 3.0],
                [1.1, 2.0, 3.0],
                [1.0, 2.1, 3.0],
            ]
        ),
    )
    np.save(results / f"{timestamp}.npy", np.array([1.05, 2.05, 3.0]))
    np.save(truth_dir / f"{timestamp}.npy", np.array([1.0, 2.0, 3.0]))

    discovered = discover_sequence_paths(tmp_path)
    candidates, truth, _ = load_sequence_export(
        discovered[0],
        voxel_size_m=0.5,
        min_cluster_points=3,
    )

    assert [sequence.root.relative_to(tmp_path).as_posix() for sequence in discovered] == [
        "val/fog/seq_wrapped",
    ]
    assert discovered[0].candidate_trajectory_files == (results / f"{timestamp}.npy",)
    assert discovered[0].point_cloud_files == (livox / f"{timestamp}.npy",)
    assert discovered[0].truth_file == truth_dir / f"{timestamp}.npy"
    assert candidates.rows["source"].tolist() == ["livox_avia", "tracking_results"]
    assert candidates.rows["time_s"].tolist() == [20.0, 20.0]
    assert truth is not None
    assert truth.rows["time_s"].tolist() == [20.0]


def test_cli_accepts_explicit_numpy_candidate_and_truth_files(tmp_path: Path) -> None:
    candidates = tmp_path / "trajectory.npy"
    truth = tmp_path / "truth.npy"
    output = tmp_path / "out"
    rows = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, 0.0, 1.0],
            [3.0, 3.0, 0.0, 1.0],
        ]
    )
    np.save(candidates, rows)
    np.save(truth, rows)

    status = mmuad_cli_main(
        [
            "--candidate-file",
            str(candidates),
            "--truth-file",
            str(truth),
            "--output-dir",
            str(output),
            "--submission-csv",
            str(output / "submission.csv"),
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output / "mmuad_estimates.csv")
    assert estimates["time_s"].tolist() == [0.0, 1.0, 2.0, 3.0]
    metrics = json.loads((output / "mmuad_metrics.json").read_text(encoding="utf-8"))
    assert metrics["pooled"]["mean_3d_m"] < 1.0
    assert (output / "submission.csv").exists()


def test_layout_inspector_reports_modalities_and_missing_fields(tmp_path: Path) -> None:
    seq = tmp_path / "seq001"
    seq.mkdir()
    (seq / "calibration.json").write_text(json.dumps({"sensors": {}}), encoding="utf-8")
    (seq / "camera_0001.png").write_bytes(b"not-a-real-image")
    (seq / "radar_12.5.csv").write_text("x_m,y_m,z_m\n1,2,3\n", encoding="utf-8")
    (seq / "lidar_12.5.pcd").write_text(
        "\n".join(
            [
                "VERSION 0.7",
                "FIELDS x y z",
                "SIZE 4 4 4",
                "TYPE F F F",
                "COUNT 1 1 1",
                "WIDTH 1",
                "HEIGHT 1",
                "POINTS 1",
                "DATA ascii",
                "0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    report = inspect_sequence_root(tmp_path)
    assert report["sequence_count"] == 1
    assert report["category_counts"]["image"] == 1
    assert report["category_counts"]["point_cloud"] == 1
    assert report["modality_counts"]["camera"] == 1
    assert report["modality_counts"]["lidar"] >= 1
    assert "truth" in report["sequences"][0]["missing_for_tracking_smoke"]
    json_path = tmp_path / "layout.json"
    csv_path = tmp_path / "layout.csv"
    write_layout_report(report, json_path=json_path, csv_path=csv_path)
    assert json_path.exists()
    assert csv_path.exists()


def test_layout_inspectors_classify_topic_maps(tmp_path: Path) -> None:
    exported = tmp_path / "seq_exported_topic_map"
    exported.mkdir()
    (exported / "topic_map.json").write_text(
        json.dumps(
            {
                "sequence_id": "seq_exported_topic_map",
                "exports": [
                    {
                        "kind": "candidate",
                        "path": "detections.csv",
                    },
                    {
                        "kind": "pose_truth",
                        "path": "truth_export.csv",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (exported / "detections.csv").write_text("time_s,x_m,y_m,z_m\n0,0,0,1\n", encoding="utf-8")
    exported_yaml = tmp_path / "seq_exported_yaml_topic_map"
    exported_yaml.mkdir()
    (exported_yaml / "topic_map.yaml").write_text(
        "\n".join(
            [
                "sequence_id: seq_exported_yaml_topic_map",
                "exports:",
                "  - kind: candidate",
                "    path: detections.csv",
                "  - kind: pose_truth",
                "    path: truth_export.csv",
            ]
        ),
        encoding="utf-8",
    )
    (exported_yaml / "detections.csv").write_text(
        "time_s,x_m,y_m,z_m\n0,0,0,1\n",
        encoding="utf-8",
    )
    native = tmp_path / "seq_native_topic_map"
    native.mkdir()
    (native / "topic_map_native.json").write_text(
        json.dumps(
            {
                "sequence_id": "seq_native_topic_map",
                "exports": [
                    {
                        "topic": "/radar/points",
                        "kind": "pointcloud2_candidate",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    detailed = inspect_sequence_root(tmp_path)
    detailed_by_name = {row["relative_path"]: row for row in detailed["files"]}
    assert detailed_by_name["topic_map.json"]["category"] == "topic_map_export"
    assert detailed_by_name["topic_map.json"]["topic_map_has_truth_export"] is True
    assert detailed_by_name["topic_map.yaml"]["category"] == "topic_map_export"
    assert detailed_by_name["topic_map.yaml"]["topic_map_has_truth_export"] is True
    assert detailed["category_counts"]["topic_map_native"] == 1
    by_sequence = {
        row["sequence_id"]: row["missing_for_tracking_smoke"]
        for row in detailed["sequences"]
    }
    assert "truth" not in by_sequence["seq_exported_topic_map"]
    assert "candidate_or_point_cloud" not in by_sequence["seq_exported_topic_map"]
    assert "truth" not in by_sequence["seq_exported_yaml_topic_map"]
    assert "candidate_or_point_cloud" in by_sequence["seq_native_topic_map"]

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["topic_map_export"] == 2
    assert inventory["category_counts"]["topic_map_native"] == 1
    exported_summary = next(
        row
        for row in inventory["sequence_candidates"]
        if row["sequence_id"] == "seq_exported_topic_map"
    )
    native_summary = next(
        row
        for row in inventory["sequence_candidates"]
        if row["sequence_id"] == "seq_native_topic_map"
    )
    exported_yaml_summary = next(
        row
        for row in inventory["sequence_candidates"]
        if row["sequence_id"] == "seq_exported_yaml_topic_map"
    )
    assert exported_summary["has_topic_map_export"] is True
    assert exported_summary["has_candidates_or_points"] is True
    assert exported_summary["has_truth_or_labels"] is True
    assert exported_yaml_summary["has_topic_map_export"] is True
    assert exported_yaml_summary["has_truth_or_labels"] is True
    assert native_summary["has_native_topic_map"] is True
    assert native_summary["has_topic_map_export"] is False
    assert any("Exported topic-map" in item for item in inventory["recommendations"])
    assert any("Native-only topic-map" in item for item in inventory["recommendations"])


def test_layout_inspectors_classify_numpy_trajectory_exports(tmp_path: Path) -> None:
    seq = tmp_path / "seq_numpy"
    seq.mkdir()
    np.save(seq / "truth.npy", np.array([[0.0, 0.0, 0.0, 1.0]]))
    np.savez(seq / "trajectory.npz", trajectory=np.array([[0.0, 0.0, 0.0, 1.0]]))
    np.save(seq / "lidar_points.npy", np.array([[0.0, 0.0, 1.0, 0.0]]))

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}
    assert by_name["truth.npy"]["category"] == "truth"
    assert by_name["trajectory.npz"]["category"] == "candidate"
    assert by_name["lidar_points.npy"]["category"] == "point_cloud"
    assert detailed["sequences"][0]["missing_for_tracking_smoke"] == ["calibration"]

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["truth_or_label"] == 1
    assert inventory["category_counts"]["candidate_or_point_table"] == 2
    sequence = inventory["sequence_candidates"][0]
    assert sequence["has_truth_or_labels"] is True
    assert sequence["has_candidates_or_points"] is True


def test_layout_inspectors_classify_json_table_exports(tmp_path: Path) -> None:
    seq = tmp_path / "seq_json_tables"
    seq.mkdir()
    (seq / "candidates.json").write_text(
        json.dumps({"candidates": [{"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0}]}),
        encoding="utf-8",
    )
    (seq / "truth.json").write_text(
        json.dumps({"truth": [{"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0}]}),
        encoding="utf-8",
    )
    (seq / "lidar_points.json").write_text(
        json.dumps({"points": [{"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0}]}),
        encoding="utf-8",
    )
    (seq / "classes.json").write_text(
        json.dumps({"seq_json_tables": "quadrotor"}),
        encoding="utf-8",
    )

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}
    assert by_name["candidates.json"]["category"] == "candidate"
    assert by_name["truth.json"]["category"] == "truth"
    assert by_name["lidar_points.json"]["category"] == "point_cloud_csv"
    assert by_name["classes.json"]["category"] == "class_label"
    assert detailed["sequences"][0]["missing_for_tracking_smoke"] == ["calibration"]

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["candidate_or_point_table"] == 2
    assert inventory["category_counts"]["truth_or_label"] == 1
    assert inventory["category_counts"]["class_or_label"] == 1
    sequence = inventory["sequence_candidates"][0]
    assert sequence["has_candidates_or_points"] is True
    assert sequence["has_truth_or_labels"] is True
    assert sequence["has_class_labels"] is True


def test_layout_inspectors_classify_yaml_class_exports(tmp_path: Path) -> None:
    seq = tmp_path / "seq_yaml_classes"
    seq.mkdir()
    (seq / "classes.yaml").write_text(
        "\n".join(
            [
                "class_map:",
                "  seq_yaml_classes:",
                "    uav_type: quadrotor",
            ]
        ),
        encoding="utf-8",
    )

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}
    assert by_name["classes.yaml"]["category"] == "class_label"

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["class_or_label"] == 1
    assert inventory["sequence_candidates"][0]["has_class_labels"] is True


def test_layout_inspectors_classify_yaml_camera_intrinsics(tmp_path: Path) -> None:
    seq = tmp_path / "seq_camera_yaml"
    camera = seq / "cam0"
    camera.mkdir(parents=True)
    (camera / "camera_info.yaml").write_text("{}", encoding="utf-8")
    (seq / "intrinsics.yml").write_text("{}", encoding="utf-8")

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}
    assert by_name["cam0/camera_info.yaml"]["category"] == "calibration"
    assert by_name["intrinsics.yml"]["category"] == "calibration"
    assert "calibration" not in detailed["sequences"][0]["missing_for_tracking_smoke"]

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["calibration"] == 2
    assert inventory["sequence_candidates"][0]["has_calibration"] is True


def test_layout_inspectors_classify_jsonl_table_exports(tmp_path: Path) -> None:
    seq = tmp_path / "seq_jsonl_tables"
    seq.mkdir()
    (seq / "candidates.jsonl").write_text(
        json.dumps({"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0}),
        encoding="utf-8",
    )
    (seq / "truth.ndjson").write_text(
        json.dumps({"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0}),
        encoding="utf-8",
    )
    (seq / "lidar_points.jsonl").write_text(
        json.dumps({"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0}),
        encoding="utf-8",
    )
    (seq / "classes.ndjson").write_text(
        json.dumps({"sequence_id": "seq_jsonl_tables", "uav_type": "quadrotor"}),
        encoding="utf-8",
    )

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}
    assert by_name["candidates.jsonl"]["category"] == "candidate"
    assert by_name["truth.ndjson"]["category"] == "truth"
    assert by_name["lidar_points.jsonl"]["category"] == "point_cloud_csv"
    assert by_name["classes.ndjson"]["category"] == "class_label"

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["candidate_or_point_table"] == 2
    assert inventory["category_counts"]["truth_or_label"] == 1
    assert inventory["category_counts"]["class_or_label"] == 1


def test_layout_inspectors_classify_gzipped_table_exports(tmp_path: Path) -> None:
    seq = tmp_path / "seq_gzip_tables"
    seq.mkdir()
    pd.DataFrame(
        {"time_s": [0.0], "x_m": [0.0], "y_m": [0.0], "z_m": [1.0]}
    ).to_csv(seq / "candidates.csv.gz", index=False, compression="gzip")
    with gzip.open(seq / "truth.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0}))
    with gzip.open(seq / "lidar_points.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 1.0}))
    with gzip.open(seq / "classes.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"sequence_id": "seq_gzip_tables", "uav_type": "quadrotor"}))

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}
    assert by_name["candidates.csv.gz"]["category"] == "candidate"
    assert by_name["truth.jsonl.gz"]["category"] == "truth"
    assert by_name["lidar_points.jsonl.gz"]["category"] == "point_cloud_csv"
    assert by_name["classes.jsonl.gz"]["category"] == "class_label"

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["candidate_or_point_table"] == 2
    assert inventory["category_counts"]["truth_or_label"] == 1
    assert inventory["category_counts"]["class_or_label"] == 1


def test_layout_inspectors_classify_mmuad_modality_folders(tmp_path: Path) -> None:
    seq = tmp_path / "seq_foldered"
    (seq / "livox_avia").mkdir(parents=True)
    (seq / "ground_truth").mkdir()
    (seq / "tracking_results").mkdir()
    (seq / "class").mkdir()
    np.save(seq / "livox_avia" / "20.0.npy", np.zeros((3, 3)))
    np.asarray([[0.0, 0.0, 1.0, 0.25]], dtype="<f4").tofile(
        seq / "livox_avia" / "20.1.bin"
    )
    np.save(seq / "ground_truth" / "20.0.npy", np.array([0.0, 0.0, 1.0]))
    np.save(seq / "tracking_results" / "20.0.npy", np.array([0.0, 0.0, 1.0]))
    np.save(seq / "class" / "20.0.npy", np.array(2))

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}

    assert by_name["livox_avia/20.0.npy"]["category"] == "point_cloud"
    assert by_name["livox_avia/20.1.bin"]["category"] == "point_cloud"
    assert abs(float(by_name["livox_avia/20.1.bin"]["inferred_time_s"]) - 20.1) < 1e-12
    assert by_name["ground_truth/20.0.npy"]["category"] == "truth"
    assert by_name["tracking_results/20.0.npy"]["category"] == "candidate"
    assert by_name["class/20.0.npy"]["category"] == "class_label"
    assert detailed["sequences"][0]["missing_for_tracking_smoke"] == ["calibration"]

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["point_cloud"] == 1
    assert inventory["category_counts"]["truth_or_label"] == 1
    assert inventory["category_counts"]["class_or_label"] == 1
    assert inventory["category_counts"]["candidate_or_point_table"] == 2
    sequence = inventory["sequence_candidates"][0]
    assert sequence["has_truth_or_labels"] is True
    assert sequence["has_candidates_or_points"] is True
    assert sequence["has_class_labels"] is True


def test_layout_inspectors_classify_audio_streams_without_candidate_readiness(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_audio"
    audio = seq / "microphone"
    audio.mkdir(parents=True)
    (audio / "mic_12.5.wav").write_bytes(b"RIFF-not-real-audio-but-counted")

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}

    assert by_name["microphone/mic_12.5.wav"]["category"] == "audio"
    assert by_name["microphone/mic_12.5.wav"]["modality"] == "audio"
    assert abs(float(by_name["microphone/mic_12.5.wav"]["inferred_time_s"]) - 12.5) < 1e-12
    assert detailed["sequences"][0]["missing_for_tracking_smoke"] == [
        "truth",
        "calibration",
        "candidate_or_point_cloud",
    ]

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["audio"] == 1
    sequence = inventory["sequence_candidates"][0]
    assert sequence["has_candidates_or_points"] is False
    assert any("Audio files" in item for item in inventory["recommendations"])


def test_layout_inspectors_preserve_sequence_ids_under_split_folders(tmp_path: Path) -> None:
    seq = tmp_path / "val" / "seq0001"
    (seq / "livox_avia").mkdir(parents=True)
    (seq / "ground_truth").mkdir()
    (seq / "class").mkdir()
    np.save(seq / "livox_avia" / "20.0.npy", np.zeros((3, 3)))
    np.save(seq / "ground_truth" / "20.0.npy", np.array([0.0, 0.0, 1.0]))
    np.save(seq / "class" / "20.0.npy", np.array(2))

    detailed = inspect_sequence_root(tmp_path)
    assert [row["sequence_id"] for row in detailed["sequences"]] == ["seq0001"]
    by_name = {row["relative_path"]: row for row in detailed["files"]}
    assert by_name["livox_avia/20.0.npy"]["sequence_id"] == "seq0001"

    inventory = inspect_mmuad_layout(tmp_path)
    assert [row["sequence_id"] for row in inventory["sequence_candidates"]] == ["seq0001"]
    sequence = inventory["sequence_candidates"][0]
    assert sequence["has_candidates_or_points"] is True
    assert sequence["has_truth_or_labels"] is True
    assert sequence["has_class_labels"] is True


def test_layout_inspectors_inventory_zip_archive_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "mmuad_export.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "val/seq001/topic_map.json",
            json.dumps(
                {
                    "sequence_id": "seq001",
                    "exports": [
                        {"kind": "candidate", "path": "candidates.csv"},
                        {"kind": "pose_truth", "path": "truth.csv"},
                    ],
                }
            ),
        )
        archive.writestr("val/seq001/candidates.csv", "time_s,x_m,y_m,z_m\n0,0,0,1\n")
        archive.writestr("val/seq001/truth.csv", "time_s,x_m,y_m,z_m\n0,0,0,1\n")
        archive.writestr(
            "val/seq001/livox_avia/20.5.pcd",
            "\n".join(
                [
                    "VERSION 0.7",
                    "FIELDS x y z",
                    "SIZE 4 4 4",
                    "TYPE F F F",
                    "COUNT 1 1 1",
                    "WIDTH 1",
                    "HEIGHT 1",
                    "POINTS 1",
                    "DATA ascii",
                    "0 0 0",
                ]
            ),
        )

    detailed = inspect_sequence_root(archive_path)
    assert detailed["archive_count"] == 1
    assert detailed["archive_member_count"] == 4
    assert detailed["sequence_count"] == 1
    assert detailed["sequences"][0]["sequence_id"] == "seq001"
    assert detailed["sequences"][0]["missing_for_tracking_smoke"] == ["calibration"]
    by_name = {
        row["relative_path"].split("::", 1)[1]: row
        for row in detailed["files"]
    }
    assert by_name["val/seq001/topic_map.json"]["category"] == "topic_map_export"
    assert by_name["val/seq001/topic_map.json"]["topic_map_has_truth_export"] is True
    assert by_name["val/seq001/candidates.csv"]["category"] == "candidate"
    assert by_name["val/seq001/truth.csv"]["category"] == "truth"
    assert by_name["val/seq001/livox_avia/20.5.pcd"]["category"] == "point_cloud"
    assert by_name["val/seq001/livox_avia/20.5.pcd"]["inferred_time_s"] == 20.5

    inventory = inspect_mmuad_layout(archive_path)
    assert inventory["archive_count"] == 1
    assert inventory["archives"][0]["path"] == "mmuad_export.zip"
    assert inventory["category_counts"]["topic_map_export"] == 1
    assert inventory["category_counts"]["candidate_or_point_table"] == 1
    assert inventory["category_counts"]["truth_or_label"] == 1
    assert inventory["category_counts"]["point_cloud"] == 1
    sequence = inventory["sequence_candidates"][0]
    assert sequence["sequence_id"] == "seq001"
    assert sequence["has_topic_map_export"] is True
    assert sequence["has_candidates_or_points"] is True
    assert sequence["has_truth_or_labels"] is True
    assert any("Archive files found" in item for item in inventory["recommendations"])


def test_layout_inspectors_inventory_tar_archive_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "mmuad_raw.tar.gz"

    def add_text(archive: tarfile.TarFile, name: str, text: str) -> None:
        payload = text.encode("utf-8")
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with tarfile.open(archive_path, "w:gz") as archive:
        add_text(
            archive,
            "train/seq002/topic_map_native.yaml",
            "\n".join(
                [
                    "sequence_id: seq002",
                    "exports:",
                    "  - topic: /radar/points",
                    "    kind: pointcloud2_candidate",
                ]
            ),
        )
        add_text(archive, "train/seq002/recording.mcap", "not-a-real-recording")
        add_text(archive, "train/seq002/classes.yaml", "seq002: 1\n")

    detailed = inspect_sequence_root(tmp_path)
    assert detailed["archive_count"] == 1
    assert detailed["archive_member_count"] == 3
    by_name = {
        row["relative_path"].split("::", 1)[1]: row
        for row in detailed["files"]
    }
    assert by_name["train/seq002/topic_map_native.yaml"]["category"] == "topic_map_native"
    assert by_name["train/seq002/recording.mcap"]["category"] == "ros_recording"
    assert by_name["train/seq002/classes.yaml"]["category"] == "class_label"

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["archives"][0]["format"] == "tar"
    assert inventory["category_counts"]["topic_map_native"] == 1
    assert inventory["category_counts"]["rosbag_or_recording"] == 1
    assert inventory["category_counts"]["class_or_label"] == 1
    sequence = inventory["sequence_candidates"][0]
    assert sequence["sequence_id"] == "seq002"
    assert sequence["has_native_topic_map"] is True
    assert sequence["has_candidates_or_points"] is True


def test_cli_inspect_layout_only_inventories_archive_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "mmuad_cli_export.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("seq_cli/candidates.csv", "time_s,x_m,y_m,z_m\n0,0,0,1\n")
        archive.writestr("seq_cli/truth.csv", "time_s,x_m,y_m,z_m\n0,0,0,1\n")

    output = tmp_path / "layout_out"
    status = mmuad_cli_main(
        [
            "--sequence-root",
            str(tmp_path),
            "--inspect-layout-only",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    report = json.loads((output / "mmuad_layout_report.json").read_text(encoding="utf-8"))
    assert report["archive_count"] == 1
    assert report["archive_member_count"] == 2
    assert report["sequence_candidates"][0]["sequence_id"] == "seq_cli"
    assert report["sequence_candidates"][0]["has_truth_or_labels"] is True


def test_submission_evaluator_matches_truth(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 1.0],
            "track_id": ["uav", "uav"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
            "score": [1.0, 1.0],
        }
    ).to_csv(submission, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
        }
    ).to_csv(truth, index=False)
    metrics = evaluate_submission_csv(submission, truth)
    assert metrics["official_ug2_metric"] is False
    assert metrics["pooled"]["matched_count"] == 2
    assert metrics["pooled"]["mean_3d_m"] == 0.0


def test_submission_evaluator_supports_public_track5_protocol(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "Sequence": ["s1", "s1"],
            "Timestamp": [0.0, 1.0],
            "Track": ["uav", "uav"],
            "x": [0.0, 2.0],
            "y": [0.0, 0.0],
            "z": [2.0, 2.0],
            "Classification": [2, 2],
        }
    ).to_csv(submission, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
            "classification": [2, 2],
        }
    ).to_csv(truth, index=False)

    metrics = evaluate_submission_csv(
        submission,
        truth,
        metric_protocol="public-track5",
        timestamp_tolerance_s=0.0,
    )

    assert metrics["schema"] == "raft-uav-mmuad-submission-eval-v1"
    assert metrics["metric_protocol"] == "public_track5_timestamp_aligned"
    assert metrics["public_track5_metric"] is True
    assert metrics["official_ug2_metric"] is False
    assert metrics["closed_codabench_evaluator"] is False
    assert metrics["stable_submission_csv"] is True
    assert metrics["score_valid_for_leaderboard"] is True
    assert metrics["leaderboard_ready"] is False
    assert metrics["codabench_upload_ready"] is False
    assert (
        "stable_submission_csv_not_official_track5_package"
        in metrics["leaderboard_blocking_reasons"]
    )
    assert metrics["matched_count"] == 2
    assert metrics["missing_prediction_count"] == 0
    assert metrics["extra_prediction_count"] == 0
    assert metrics["pooled"]["pose_mse_loss_m2"] == 0.5
    assert metrics["pooled"]["classification_accuracy"] == 1.0


def test_cli_evaluates_submission_csv_with_public_track5_protocol(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission.csv"
    truth = tmp_path / "truth.csv"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 1.0],
            "track_id": ["uav", "uav"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
            "classification": [4, 4],
        }
    ).to_csv(submission, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
            "classification": [4, 4],
        }
    ).to_csv(truth, index=False)

    status = mmuad_cli_main(
        [
            "--evaluate-submission-csv",
            str(submission),
            "--evaluate-truth-csv",
            str(truth),
            "--evaluation-protocol",
            "public-track5",
            "--evaluation-timestamp-tolerance-s",
            "0",
            "--evaluation-json",
            str(output / "submission_public_track5.json"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    metrics = json.loads(
        (output / "submission_public_track5.json").read_text(encoding="utf-8")
    )
    assert metrics["metric_protocol"] == "public_track5_timestamp_aligned"
    assert metrics["stable_submission_csv"] is True
    assert metrics["score_valid_for_leaderboard"] is True
    assert metrics["leaderboard_ready"] is False
    assert metrics["pooled"]["pose_mse_loss_m2"] == 0.0
    assert metrics["pooled"]["classification_accuracy"] == 1.0


def test_submission_evaluator_accepts_nanosecond_timestamps(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "timestamp_ns": [1_000_000_000],
            "track_id": ["uav"],
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [2.0],
            "score": [1.0],
        }
    ).to_csv(submission, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "time_s": [1.0],
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [2.0],
        }
    ).to_csv(truth, index=False)

    metrics = evaluate_submission_csv(submission, truth)

    assert metrics["pooled"]["matched_count"] == 1
    assert metrics["pooled"]["mean_3d_m"] == 0.0


def test_submission_evaluator_accepts_numpy_truth_file(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    truth = tmp_path / "truth.npy"
    pd.DataFrame(
        {
            "sequence_id": ["default", "default"],
            "time_s": [0.0, 1.0],
            "track_id": ["uav", "uav"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [2.0, 2.0],
            "score": [1.0, 1.0],
        }
    ).to_csv(submission, index=False)
    np.save(truth, np.array([[0.0, 0.0, 0.0, 2.0], [1.0, 1.0, 0.0, 2.0]]))

    metrics = evaluate_submission_csv(submission, truth)

    assert metrics["pooled"]["matched_count"] == 2
    assert metrics["pooled"]["mean_3d_m"] == 0.0


def test_mmaud_results_local_evaluator_matches_truth(tmp_path: Path) -> None:
    results = tmp_path / "mmaud_results.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "timestamp": [0.0, 1.0],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [10.0, 10.0],
            "uav_type": ["Mavic3", "Mavic3"],
            "score": [1.0, 1.0],
        }
    ).to_csv(results, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
        }
    ).to_csv(truth, index=False)
    evaluated = evaluate_mmaud_results(
        load_mmaud_results_csv(results),
        load_truth_csv(truth),
    )
    assert evaluated["summary"]["matched_count"] == 2
    assert evaluated["summary"]["pooled"]["max_3d_m"] == 0.0


def test_cli_evaluates_results_with_numpy_truth_file(tmp_path: Path) -> None:
    results = tmp_path / "mmaud_results.csv"
    truth = tmp_path / "truth.npy"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["default", "default"],
            "timestamp": [0.0, 1.0],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [10.0, 10.0],
            "uav_type": ["Mavic3", "Mavic3"],
            "score": [1.0, 1.0],
        }
    ).to_csv(results, index=False)
    np.save(truth, np.array([[0.0, 0.0, 0.0, 10.0], [1.0, 1.0, 0.0, 10.0]]))

    status = mmuad_cli_main(
        [
            "--evaluate-results-csv",
            str(results),
            "--evaluate-truth-file",
            str(truth),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    metrics = json.loads((output / "mmuad_local_evaluation.json").read_text(encoding="utf-8"))
    assert metrics["matched_count"] == 2
    assert metrics["pooled"]["mean_3d_m"] == 0.0


def test_cli_evaluates_results_with_class_map_file_alias(tmp_path: Path) -> None:
    results = tmp_path / "mmaud_results.csv"
    truth = tmp_path / "truth.csv"
    class_map = tmp_path / "classes.yaml"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "timestamp": [0.0, 1.0],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [10.0, 10.0],
            "uav_type": ["Mavic3", "Mavic3"],
            "score": [1.0, 1.0],
        }
    ).to_csv(results, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
        }
    ).to_csv(truth, index=False)
    class_map.write_text(
        "\n".join(["class_map:", "  seqA:", "    uav_type: Mavic3"]),
        encoding="utf-8",
    )

    status = mmuad_cli_main(
        [
            "--evaluate-results-csv",
            str(results),
            "--evaluate-truth-csv",
            str(truth),
            "--evaluation-class-map-file",
            str(class_map),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    metrics = json.loads((output / "mmuad_local_evaluation.json").read_text(encoding="utf-8"))
    assert metrics["matched_count"] == 2
    assert metrics["pooled"]["uav_type_accuracy"] == 1.0


def test_cli_evaluates_ug2_codabench_zip(tmp_path: Path) -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [10.0, 10.0],
        }
    )
    zip_path = write_ug2_codabench_zip(
        estimates,
        tmp_path / "ug2_submission.zip",
        class_name="Mavic3",
    )
    truth = tmp_path / "truth.csv"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "uav_type": ["Mavic3", "Mavic3"],
        }
    ).to_csv(truth, index=False)

    status = mmuad_cli_main(
        [
            "--evaluate-results-zip",
            str(zip_path),
            "--evaluate-truth-csv",
            str(truth),
            "--output-dir",
            str(output),
            "--evaluation-json",
            str(output / "eval.json"),
        ]
    )

    assert status == 0
    metrics = json.loads((output / "eval.json").read_text(encoding="utf-8"))
    assert metrics["matched_count"] == 2
    assert metrics["pooled"]["pose_mse_loss_m2"] == 0.0
    assert metrics["pooled"]["uav_type_accuracy"] == 1.0


def test_cli_evaluates_official_zip_with_public_track5_protocol(tmp_path: Path) -> None:
    zip_path = tmp_path / "official_submission.zip"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(2,0,10)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", official.to_csv(index=False))
    truth = tmp_path / "truth.csv"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "uav_type": ["2", "1"],
        }
    ).to_csv(truth, index=False)

    status = mmuad_cli_main(
        [
            "--evaluate-results-zip",
            str(zip_path),
            "--evaluate-truth-csv",
            str(truth),
            "--evaluation-protocol",
            "public-track5",
            "--evaluation-timestamp-tolerance-s",
            "0",
            "--evaluation-json",
            str(output / "public_track5_eval.json"),
            "--evaluation-rows-csv",
            str(output / "public_track5_eval_rows.csv"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    metrics = json.loads((output / "public_track5_eval.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(output / "public_track5_eval_rows.csv")
    assert metrics["metric_protocol"] == "public_track5_timestamp_aligned"
    assert metrics["truth_count"] == 2
    assert metrics["matched_count"] == 2
    assert metrics["leaderboard_ready"] is True
    assert metrics["score_valid_for_leaderboard"] is True
    assert metrics["codabench_upload_ready"] is True
    assert metrics["pooled"]["mean_square_loss_m2"] == 0.5
    assert metrics["pooled"]["classification_accuracy"] == 0.5
    assert rows["matched"].tolist() == [True, True]


def test_cli_evaluates_nested_official_zip_but_marks_not_upload_ready(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "nested_official_submission.zip"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(1,0,10)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("submission/mmaud_results.csv", official.to_csv(index=False))
        archive.writestr("README.txt", "not upload-ready until normalized")
    truth = tmp_path / "truth.csv"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "uav_type": ["2", "2"],
        }
    ).to_csv(truth, index=False)

    status = mmuad_cli_main(
        [
            "--evaluate-results-zip",
            str(zip_path),
            "--evaluate-truth-csv",
            str(truth),
            "--evaluation-protocol",
            "public-track5",
            "--evaluation-timestamp-tolerance-s",
            "0",
            "--evaluation-json",
            str(output / "public_track5_eval_nested.json"),
            "--evaluation-rows-csv",
            str(output / "public_track5_eval_nested_rows.csv"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    metrics = json.loads(
        (output / "public_track5_eval_nested.json").read_text(encoding="utf-8")
    )
    rows = pd.read_csv(output / "public_track5_eval_nested_rows.csv")
    assert metrics["matched_count"] == 2
    assert metrics["pooled"]["mean_square_loss_m2"] == 0.0
    assert rows["matched"].tolist() == [True, True]
    assert metrics["official_submission_valid"] is False
    assert metrics["score_valid_for_leaderboard"] is False
    assert metrics["leaderboard_ready"] is False
    assert metrics["codabench_upload_ready"] is False
    validation = metrics["official_submission_validation"]
    assert validation["nested_mmaud_results_csv_members"] == [
        "submission/mmaud_results.csv"
    ]
    assert validation["has_root_mmaud_results_csv"] is False
    assert "official_upload_package_not_ready" in metrics["leaderboard_blocking_reasons"]
    assert "official_zip_members_invalid" in metrics["leaderboard_blocking_reasons"]


def test_cli_public_track5_evaluation_reports_csv_not_upload_ready(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "mmaud_results.csv"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(1,0,10)"],
            "Classification": [2, 2],
        }
    )
    official.to_csv(results_csv, index=False)
    truth = tmp_path / "truth.csv"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "uav_type": ["2", "2"],
        }
    ).to_csv(truth, index=False)

    status = mmuad_cli_main(
        [
            "--evaluate-results-csv",
            str(results_csv),
            "--evaluate-truth-csv",
            str(truth),
            "--evaluation-protocol",
            "public-track5",
            "--evaluation-timestamp-tolerance-s",
            "0",
            "--evaluation-json",
            str(output / "public_track5_eval_csv.json"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    metrics = json.loads(
        (output / "public_track5_eval_csv.json").read_text(encoding="utf-8")
    )
    assert metrics["matched_count"] == 2
    assert metrics["score_valid_for_leaderboard"] is True
    assert metrics["official_submission_valid"] is True
    assert metrics["codabench_upload_ready"] is False
    assert metrics["leaderboard_ready"] is False
    assert metrics["leaderboard_blocking_reasons"] == [
        "official_upload_package_not_ready"
    ]
    assert metrics["official_submission_validation"]["is_zip"] is False

    with pytest.raises(SystemExit, match="official_upload_package_not_ready"):
        mmuad_cli_main(
            [
                "--evaluate-results-csv",
                str(results_csv),
                "--evaluate-truth-csv",
                str(truth),
                "--evaluation-protocol",
                "public-track5",
                "--evaluation-timestamp-tolerance-s",
                "0",
                "--evaluation-require-complete-track5",
                "--evaluation-json",
                str(output / "public_track5_eval_csv_require.json"),
                "--output-dir",
                str(output),
            ]
        )


def test_cli_evaluates_official_zip_against_official_truth_zip(
    tmp_path: Path,
) -> None:
    predictions_zip = tmp_path / "official_submission.zip"
    truth_zip = tmp_path / "official_truth.zip"
    output = tmp_path / "out"
    predictions = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(2,0,10)"],
            "Classification": [2, 1],
        }
    )
    truth = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(1,0,10)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(predictions_zip, "w") as archive:
        archive.writestr("mmaud_results.csv", predictions.to_csv(index=False))
    with ZipFile(truth_zip, "w") as archive:
        archive.writestr("mmaud_results.csv", truth.to_csv(index=False))

    status = mmuad_cli_main(
        [
            "--evaluate-results-zip",
            str(predictions_zip),
            "--evaluate-truth-file",
            str(truth_zip),
            "--evaluation-protocol",
            "public-track5",
            "--evaluation-timestamp-tolerance-s",
            "0",
            "--evaluation-json",
            str(output / "official_truth_eval.json"),
            "--evaluation-rows-csv",
            str(output / "official_truth_eval_rows.csv"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    metrics = json.loads(
        (output / "official_truth_eval.json").read_text(encoding="utf-8")
    )
    rows = pd.read_csv(output / "official_truth_eval_rows.csv")
    assert metrics["leaderboard_ready"] is True
    assert metrics["score_valid_for_leaderboard"] is True
    assert metrics["codabench_upload_ready"] is True
    assert metrics["pooled"]["mean_square_loss_m2"] == 0.5
    assert metrics["pooled"]["classification_accuracy"] == 0.5
    assert rows["truth_x_m"].tolist() == [0.0, 1.0]
    assert rows["truth_uav_type"].tolist() == [2, 2]


def test_cli_evaluates_official_zip_against_nested_official_truth_zip(
    tmp_path: Path,
) -> None:
    predictions_zip = tmp_path / "official_submission.zip"
    truth_zip = tmp_path / "nested_official_truth.zip"
    output = tmp_path / "out"
    predictions = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(2,0,10)"],
            "Classification": [2, 1],
        }
    )
    truth = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(1,0,10)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(predictions_zip, "w") as archive:
        archive.writestr("mmaud_results.csv", predictions.to_csv(index=False))
    with ZipFile(truth_zip, "w") as archive:
        archive.writestr("truth/mmaud_results.csv", truth.to_csv(index=False))
        archive.writestr("README.txt", "truth bundle exported from a nested folder")

    status = mmuad_cli_main(
        [
            "--evaluate-results-zip",
            str(predictions_zip),
            "--evaluate-truth-file",
            str(truth_zip),
            "--evaluation-protocol",
            "public-track5",
            "--evaluation-timestamp-tolerance-s",
            "0",
            "--evaluation-json",
            str(output / "nested_official_truth_eval.json"),
            "--evaluation-rows-csv",
            str(output / "nested_official_truth_eval_rows.csv"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    metrics = json.loads(
        (output / "nested_official_truth_eval.json").read_text(encoding="utf-8")
    )
    rows = pd.read_csv(output / "nested_official_truth_eval_rows.csv")
    assert metrics["leaderboard_ready"] is True
    assert metrics["score_valid_for_leaderboard"] is True
    assert metrics["codabench_upload_ready"] is True
    assert metrics["pooled"]["mean_square_loss_m2"] == 0.5
    assert metrics["pooled"]["classification_accuracy"] == 0.5
    assert rows["truth_x_m"].tolist() == [0.0, 1.0]
    assert rows["truth_uav_type"].tolist() == [2, 2]


def test_cli_public_track5_evaluation_rejects_zip_with_extra_members(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "official_submission_extra.zip"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,10)", "(1,0,10)"],
            "Classification": [2, 2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", official.to_csv(index=False))
        archive.writestr("README.txt", "extra members are not public Track 5 upload shape")
    truth = tmp_path / "truth.csv"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "uav_type": ["2", "2"],
        }
    ).to_csv(truth, index=False)

    status = mmuad_cli_main(
        [
            "--evaluate-results-zip",
            str(zip_path),
            "--evaluate-truth-csv",
            str(truth),
            "--evaluation-protocol",
            "public-track5",
            "--evaluation-timestamp-tolerance-s",
            "0",
            "--evaluation-json",
            str(output / "public_track5_eval.json"),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    metrics = json.loads((output / "public_track5_eval.json").read_text(encoding="utf-8"))
    assert metrics["matched_count"] == 2
    assert metrics["pooled"]["mean_square_loss_m2"] == 0.0
    assert metrics["official_submission_valid"] is False
    assert metrics["official_submission_validation"]["contains_only_mmaud_results_csv"] is False
    assert metrics["leaderboard_ready"] is False
    assert metrics["score_valid_for_leaderboard"] is False
    assert metrics["codabench_upload_ready"] is False
    assert "official_upload_package_not_ready" in metrics["leaderboard_blocking_reasons"]
    assert "official_zip_members_invalid" in metrics["leaderboard_blocking_reasons"]

    with pytest.raises(SystemExit, match="official_zip_members_invalid"):
        mmuad_cli_main(
            [
                "--evaluate-results-zip",
                str(zip_path),
                "--evaluate-truth-csv",
                str(truth),
                "--evaluation-protocol",
                "public-track5",
                "--evaluation-timestamp-tolerance-s",
                "0",
                "--evaluation-require-complete-track5",
                "--evaluation-json",
                str(output / "public_track5_eval_require.json"),
                "--output-dir",
                str(output),
            ]
        )


def test_cli_requires_complete_public_track5_evaluation(tmp_path: Path) -> None:
    zip_path = tmp_path / "official_submission.zip"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,10)"],
            "Classification": [2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", official.to_csv(index=False))
    truth = tmp_path / "truth.csv"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "uav_type": ["2", "2"],
        }
    ).to_csv(truth, index=False)

    with pytest.raises(SystemExit, match="not leaderboard-ready"):
        mmuad_cli_main(
            [
                "--evaluate-results-zip",
                str(zip_path),
                "--evaluate-truth-csv",
                str(truth),
                "--evaluation-protocol",
                "public-track5",
                "--evaluation-timestamp-tolerance-s",
                "0",
                "--evaluation-require-complete-track5",
                "--evaluation-json",
                str(output / "public_track5_eval.json"),
                "--evaluation-rows-csv",
                str(output / "public_track5_eval_rows.csv"),
                "--output-dir",
                str(output),
            ]
        )

    metrics = json.loads((output / "public_track5_eval.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(output / "public_track5_eval_rows.csv")
    assert metrics["leaderboard_ready"] is False
    assert metrics["score_valid_for_leaderboard"] is False
    assert "missing_predictions" in metrics["leaderboard_blocking_reasons"]
    assert "missing_prediction" in rows["unmatched_reason"].fillna("").tolist()


def test_cli_public_track5_evaluation_rejects_empty_truth_template(
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "official_submission.zip"
    official = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,10)"],
            "Classification": [2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", official.to_csv(index=False))
    truth = tmp_path / "empty_truth.csv"
    output = tmp_path / "out"
    pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"]).to_csv(
        truth,
        index=False,
    )

    with pytest.raises(SystemExit, match="no_truth_timestamps"):
        mmuad_cli_main(
            [
                "--evaluate-results-zip",
                str(zip_path),
                "--evaluate-truth-csv",
                str(truth),
                "--evaluation-protocol",
                "public-track5",
                "--evaluation-timestamp-tolerance-s",
                "0",
                "--evaluation-require-complete-track5",
                "--evaluation-json",
                str(output / "public_track5_empty_eval.json"),
                "--evaluation-rows-csv",
                str(output / "public_track5_empty_eval_rows.csv"),
                "--output-dir",
                str(output),
            ]
        )

    metrics = json.loads(
        (output / "public_track5_empty_eval.json").read_text(encoding="utf-8")
    )
    rows = pd.read_csv(output / "public_track5_empty_eval_rows.csv")
    assert metrics["truth_count"] == 0
    assert metrics["prediction_count"] == 1
    assert metrics["extra_prediction_count"] == 1
    assert metrics["truth_coverage_fraction"] == 0.0
    assert metrics["all_truth_timestamps_matched"] is False
    assert metrics["leaderboard_ready"] is False
    assert metrics["score_valid_for_leaderboard"] is False
    assert metrics["leaderboard_blocking_reasons"] == [
        "no_truth_timestamps",
        "extra_predictions",
        "official_upload_package_not_ready",
    ]
    assert rows["unmatched_reason"].tolist() == ["extra_prediction"]


def test_cli_completes_results_to_numpy_truth_template(tmp_path: Path) -> None:
    candidates = tmp_path / "trajectory.npy"
    template = tmp_path / "template_truth.npy"
    output = tmp_path / "out"
    rows = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [2.0, 2.0, 0.0, 1.0],
        ]
    )
    np.save(candidates, rows)
    np.save(template, np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]))

    status = mmuad_cli_main(
        [
            "--candidate-file",
            str(candidates),
            "--output-dir",
            str(output),
            "--complete-results-to-truth-file",
            str(template),
            "--completed-results-csv",
            str(output / "completed.csv"),
        ]
    )

    assert status == 0
    completed = pd.read_csv(output / "completed.csv")
    assert completed["timestamp"].tolist() == [0.0, 1.0]
    summary = json.loads((output / "mmuad_completion_summary.json").read_text(encoding="utf-8"))
    assert summary["requested_count"] == 2


def test_topic_map_exports_load_candidates_and_truth(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [5.0, 5.0],
        }
    ).to_csv(exports / "radar.csv", index=False)
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [5.0, 5.0],
        }
    ).to_csv(exports / "truth.csv", index=False)
    topic_map = tmp_path / "topic_map.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_ros",
                "exports": [
                    {
                        "kind": "candidate",
                        "path": "radar.csv",
                        "source": "radar",
                        "column_aliases": {
                            "stamp": "time_s",
                            "px": "x_m",
                            "py": "y_m",
                            "pz": "z_m",
                        },
                    },
                    {
                        "kind": "truth",
                        "path": "truth.csv",
                        "column_aliases": {
                            "stamp": "time_s",
                            "px": "x_m",
                            "py": "y_m",
                            "pz": "z_m",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle = load_topic_map_exports(topic_map, base_dir=exports)
    assert len(bundle.candidates.rows) == 2
    assert bundle.truth is not None
    assert len(bundle.truth.rows) == 2
    assert bundle.candidates.rows.loc[0, "sequence_id"] == "seq_ros"


def test_topic_map_exports_load_yaml_candidates_and_truth(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [5.0, 5.0],
        }
    ).to_csv(exports / "radar.csv", index=False)
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [5.0, 5.0],
        }
    ).to_csv(exports / "truth.csv", index=False)
    topic_map = tmp_path / "topic_map.yaml"
    topic_map.write_text(
        "\n".join(
            [
                "sequence_id: seq_ros_yaml",
                "exports:",
                "  - kind: candidate",
                "    path: radar.csv",
                "    source: radar",
                "    column_aliases:",
                "      stamp: time_s",
                "      px: x_m",
                "      py: y_m",
                "      pz: z_m",
                "  - kind: truth",
                "    path: truth.csv",
                "    column_aliases:",
                "      stamp: time_s",
                "      px: x_m",
                "      py: y_m",
                "      pz: z_m",
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert len(bundle.candidates.rows) == 2
    assert bundle.truth is not None
    assert len(bundle.truth.rows) == 2
    assert bundle.candidates.rows["sequence_id"].tolist() == [
        "seq_ros_yaml",
        "seq_ros_yaml",
    ]


def test_cli_runs_yaml_topic_map_file_alias(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [5.0, 5.0],
        }
    ).to_csv(exports / "radar.csv", index=False)
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "px": [0.0, 1.0],
            "py": [0.0, 0.0],
            "pz": [5.0, 5.0],
        }
    ).to_csv(exports / "truth.csv", index=False)
    topic_map = tmp_path / "topic_map.yaml"
    topic_map.write_text(
        "\n".join(
            [
                "sequence_id: seq_cli_yaml_topic_map",
                "exports:",
                "  - kind: candidate",
                "    path: radar.csv",
                "    source: radar",
                "    column_aliases:",
                "      stamp: time_s",
                "      px: x_m",
                "      py: y_m",
                "      pz: z_m",
                "  - kind: truth",
                "    path: truth.csv",
                "    column_aliases:",
                "      stamp: time_s",
                "      px: x_m",
                "      py: y_m",
                "      pz: z_m",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--topic-map-file",
            str(topic_map),
            "--topic-map-base-dir",
            str(exports),
            "--output-dir",
            str(output),
            "--submission-csv",
            str(output / "submission.csv"),
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output / "mmuad_estimates.csv")
    assert estimates["sequence_id"].tolist() == ["seq_cli_yaml_topic_map"] * 2
    assert (output / "submission.csv").exists()


def test_cli_native_ros_extraction_writes_submission_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bag = tmp_path / "seq_native.mcap"
    bag.write_bytes(b"fake native bag")
    topic_map = tmp_path / "topic_map_native.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_native",
                "exports": [{"topic": "/radar", "kind": "pose_candidate"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    extracted_dir = output / "extracted"

    def fake_extract_native_rosbag_topic_map(
        *,
        bag_path,
        topic_map_json,
        output_dir,
        voxel_size_m,
        min_points,
    ):
        assert bag_path == bag
        assert topic_map_json == topic_map
        assert voxel_size_m == 0.75
        assert min_points == 3
        output_dir.mkdir(parents=True)
        (output_dir / "native_ros_extraction_manifest.json").write_text(
            json.dumps({"schema": "fake-native-extraction"}),
            encoding="utf-8",
        )
        candidates = CandidateFrame(
            pd.DataFrame(
                {
                    "sequence_id": ["seq_native", "seq_native"],
                    "time_s": [0.0, 1.0],
                    "source": ["radar", "radar"],
                    "track_id": ["uav", "uav"],
                    "x_m": [0.0, 1.0],
                    "y_m": [0.0, 0.0],
                    "z_m": [10.0, 10.0],
                    "std_xy_m": [1.0, 1.0],
                    "std_z_m": [1.0, 1.0],
                    "confidence": [1.0, 1.0],
                    "class_name": ["2", "2"],
                }
            )
        )
        truth = TruthFrame(
            pd.DataFrame(
                {
                    "sequence_id": ["seq_native", "seq_native"],
                    "time_s": [0.0, 1.0],
                    "x_m": [0.0, 1.0],
                    "y_m": [0.0, 0.0],
                    "z_m": [10.0, 10.0],
                }
            )
        )
        return SimpleNamespace(candidates=candidates, truth=truth, manifest={})

    monkeypatch.setattr(
        "raft_uav.mmuad.cli.extract_native_rosbag_topic_map",
        fake_extract_native_rosbag_topic_map,
    )

    status = mmuad_cli_main(
        [
            "--rosbag-path",
            str(bag),
            "--topic-map-file",
            str(topic_map),
            "--native-ros-extract-output-dir",
            str(extracted_dir),
            "--output-dir",
            str(output),
            "--submission-csv",
            str(output / "submission.csv"),
            "--submission-json",
            str(output / "submission.json"),
            "--submission-zip",
            str(output / "submission.zip"),
            "--ug2-results-csv",
            str(output / "mmaud_results.csv"),
            "--ug2-codabench-zip",
            str(output / "ug2_submission.zip"),
            "--ug2-official-results-csv",
            str(output / "official_mmaud_results.csv"),
            "--ug2-official-codabench-zip",
            str(output / "official_submission.zip"),
            "--ug2-official-classification",
            "2",
            "--ug2-official-validate-on-write",
        ]
    )

    assert status == 0
    expected = [
        "mmuad_estimates.csv",
        "mmuad_metrics.json",
        "mmuad_trajectory_metrics.json",
        "submission.csv",
        "submission.json",
        "submission.zip",
        "mmaud_results.csv",
        "ug2_submission.zip",
        "official_mmaud_results.csv",
        "official_submission.zip",
        "mmuad_official_submission_validation.json",
        "mmuad_official_submission_validation_rows.csv",
        "mmuad_official_upload_manifest.json",
    ]
    for name in expected:
        assert (output / name).exists(), name
    assert (extracted_dir / "native_ros_extraction_manifest.json").exists()
    official = pd.read_csv(output / "official_mmaud_results.csv")
    assert official.columns.tolist() == [
        "Sequence",
        "Timestamp",
        "Position",
        "Classification",
    ]
    assert official["Classification"].tolist() == [2, 2]
    validation = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["valid"] is True
    assert validation["template_checked"] is True
    assert validation["template_timestamp_count"] == 2
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    manifest = json.loads(
        (output / "mmuad_official_upload_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["codabench_upload_ready"] is True
    assert manifest["leaderboard_ready"] is True


def test_cli_native_ros_extraction_uses_image_timestamps_for_official_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bag = tmp_path / "seq_native_image_template.mcap"
    bag.write_bytes(b"fake native bag")
    topic_map = tmp_path / "topic_map_native.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_native_image_template",
                "exports": [
                    {"topic": "/radar", "kind": "pose_candidate"},
                    {"topic": "/camera/image", "kind": "image_timestamps"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    extracted_dir = output / "extracted"

    def fake_extract_native_rosbag_topic_map(
        *,
        bag_path,
        topic_map_json,
        output_dir,
        voxel_size_m,
        min_points,
    ):
        assert bag_path == bag
        assert topic_map_json == topic_map
        output_dir.mkdir(parents=True)
        (output_dir / "native_ros_extraction_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "fake-native-extraction",
                    "candidate_rows": 2,
                    "truth_rows": 0,
                    "image_timestamp_rows": 2,
                }
            ),
            encoding="utf-8",
        )
        candidates = CandidateFrame(
            pd.DataFrame(
                {
                    "sequence_id": ["seq_native_image_template"] * 2,
                    "time_s": [0.0, 1.0],
                    "source": ["radar", "radar"],
                    "track_id": ["uav", "uav"],
                    "x_m": [0.0, 1.0],
                    "y_m": [0.0, 0.0],
                    "z_m": [10.0, 10.0],
                    "std_xy_m": [1.0, 1.0],
                    "std_z_m": [1.0, 1.0],
                    "confidence": [1.0, 1.0],
                    "class_name": ["2", "2"],
                }
            )
        )
        image_timestamps = pd.DataFrame(
            {
                "sequence_id": ["seq_native_image_template"] * 2,
                "time_s": [0.0, 1.0],
                "topic": ["/camera/image", "/camera/image"],
                "source": ["front", "front"],
            }
        )
        return SimpleNamespace(
            candidates=candidates,
            truth=None,
            image_timestamps=image_timestamps,
            manifest={"image_timestamp_rows": 2},
        )

    monkeypatch.setattr(
        "raft_uav.mmuad.cli.extract_native_rosbag_topic_map",
        fake_extract_native_rosbag_topic_map,
    )

    status = mmuad_cli_main(
        [
            "--rosbag-path",
            str(bag),
            "--topic-map-file",
            str(topic_map),
            "--native-ros-extract-output-dir",
            str(extracted_dir),
            "--output-dir",
            str(output),
            "--ug2-official-results-csv",
            str(output / "official_mmaud_results.csv"),
            "--ug2-official-codabench-zip",
            str(output / "official_submission.zip"),
            "--ug2-official-classification",
            "2",
            "--ug2-official-validate-on-write",
        ]
    )

    assert status == 0
    validation = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["template_checked"] is True
    assert validation["template_timestamp_count"] == 2
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True


def test_cli_native_ros_extraction_completes_official_rows_to_image_timestamps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bag = tmp_path / "seq_native_image_completion.mcap"
    bag.write_bytes(b"fake native bag")
    topic_map = tmp_path / "topic_map_native.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_native_image_completion",
                "exports": [
                    {"topic": "/radar", "kind": "pose_candidate"},
                    {"topic": "/camera/image", "kind": "image_timestamps"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    extracted_dir = output / "extracted"

    def fake_extract_native_rosbag_topic_map(
        *,
        bag_path,
        topic_map_json,
        output_dir,
        voxel_size_m,
        min_points,
    ):
        assert bag_path == bag
        assert topic_map_json == topic_map
        output_dir.mkdir(parents=True)
        (output_dir / "native_ros_extraction_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "fake-native-extraction",
                    "candidate_rows": 2,
                    "truth_rows": 0,
                    "image_timestamp_rows": 3,
                }
            ),
            encoding="utf-8",
        )
        candidates = CandidateFrame(
            pd.DataFrame(
                {
                    "sequence_id": ["seq_native_image_completion"] * 2,
                    "time_s": [0.0, 1.0],
                    "source": ["radar", "radar"],
                    "track_id": ["uav", "uav"],
                    "x_m": [0.0, 1.0],
                    "y_m": [0.0, 0.0],
                    "z_m": [10.0, 10.0],
                    "std_xy_m": [1.0, 1.0],
                    "std_z_m": [1.0, 1.0],
                    "confidence": [1.0, 1.0],
                    "class_name": ["2", "2"],
                }
            )
        )
        image_timestamps = pd.DataFrame(
            {
                "sequence_id": ["seq_native_image_completion"] * 3,
                "time_s": [0.0, 0.5, 1.0],
                "topic": ["/camera/image"] * 3,
                "source": ["front"] * 3,
            }
        )
        return SimpleNamespace(
            candidates=candidates,
            truth=None,
            image_timestamps=image_timestamps,
            manifest={"image_timestamp_rows": 3},
        )

    monkeypatch.setattr(
        "raft_uav.mmuad.cli.extract_native_rosbag_topic_map",
        fake_extract_native_rosbag_topic_map,
    )

    status = mmuad_cli_main(
        [
            "--rosbag-path",
            str(bag),
            "--topic-map-file",
            str(topic_map),
            "--native-ros-extract-output-dir",
            str(extracted_dir),
            "--output-dir",
            str(output),
            "--ug2-official-complete-to-sequence-timestamps",
            "--ug2-official-results-csv",
            str(output / "official_mmaud_results.csv"),
            "--ug2-official-codabench-zip",
            str(output / "official_submission.zip"),
            "--ug2-official-classification",
            "2",
            "--ug2-official-validate-on-write",
        ]
    )

    assert status == 0
    official = pd.read_csv(output / "official_mmaud_results.csv")
    assert official["Timestamp"].tolist() == [0.0, 0.5, 1.0]
    assert official["Classification"].tolist() == [2, 2, 2]
    completion_summary = json.loads(
        (output / "mmuad_official_timestamp_completion_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion_summary["timestamp_source"] == "native-image-timestamps"
    assert completion_summary["requested_count"] == 3
    assert completion_summary["completed_count"] == 3
    validation = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["template_checked"] is True
    assert validation["template_timestamp_count"] == 3
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True


def test_cli_sequence_root_runs_native_ros_recording(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seq = tmp_path / "seq_native_root"
    seq.mkdir()
    bag = seq / "recording.mcap"
    bag.write_bytes(b"fake native bag")
    topic_map = seq / "topic_map_native.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_native_root",
                "exports": [{"topic": "/radar", "kind": "pose_candidate"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    expected_extract_dir = output / "native_ros_extracted" / "seq_native_root"

    def fake_extract_native_rosbag_topic_map(
        *,
        bag_path,
        topic_map_json,
        output_dir,
        voxel_size_m,
        min_points,
    ):
        assert bag_path == bag
        assert topic_map_json == topic_map
        assert output_dir == expected_extract_dir
        assert voxel_size_m == 0.75
        assert min_points == 3
        output_dir.mkdir(parents=True)
        (output_dir / "native_ros_extraction_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "fake-native-extraction",
                    "candidate_rows": 2,
                    "truth_rows": 2,
                }
            ),
            encoding="utf-8",
        )
        candidates = CandidateFrame(
            pd.DataFrame(
                {
                    "sequence_id": ["seq_native_root", "seq_native_root"],
                    "time_s": [0.0, 1.0],
                    "source": ["radar", "radar"],
                    "track_id": ["uav", "uav"],
                    "x_m": [0.0, 1.0],
                    "y_m": [0.0, 0.0],
                    "z_m": [10.0, 10.0],
                    "std_xy_m": [1.0, 1.0],
                    "std_z_m": [1.0, 1.0],
                    "confidence": [1.0, 1.0],
                    "class_name": ["2", "2"],
                }
            )
        )
        truth = TruthFrame(
            pd.DataFrame(
                {
                    "sequence_id": ["seq_native_root", "seq_native_root"],
                    "time_s": [0.0, 1.0],
                    "x_m": [0.0, 1.0],
                    "y_m": [0.0, 0.0],
                    "z_m": [10.0, 10.0],
                }
            )
        )
        return SimpleNamespace(candidates=candidates, truth=truth, manifest={})

    monkeypatch.setattr(
        "raft_uav.mmuad.cli.extract_native_rosbag_topic_map",
        fake_extract_native_rosbag_topic_map,
    )

    status = mmuad_cli_main(
        [
            "--sequence-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--submission-csv",
            str(output / "submission.csv"),
            "--ug2-official-results-csv",
            str(output / "official_mmaud_results.csv"),
            "--ug2-official-codabench-zip",
            str(output / "official_submission.zip"),
            "--ug2-official-classification",
            "2",
            "--ug2-official-validate-on-write",
        ]
    )

    assert status == 0
    assert (output / "mmuad_estimates.csv").exists()
    assert (expected_extract_dir / "native_ros_extraction_manifest.json").exists()
    manifest_summary = json.loads(
        (output / "native_ros_sequence_manifests.json").read_text(encoding="utf-8")
    )
    assert manifest_summary == [
        {
            "sequence_id": "seq_native_root",
            "bag_path": str(bag),
            "topic_map_file": str(topic_map),
            "manifest_json": str(
                expected_extract_dir / "native_ros_extraction_manifest.json"
            ),
        }
    ]
    validation = json.loads(
        (output / "mmuad_official_submission_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["template_checked"] is True
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True


def test_cli_native_ros_extraction_requires_candidate_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bag = tmp_path / "seq_native_truth_only.mcap"
    bag.write_bytes(b"fake native bag")
    topic_map = tmp_path / "topic_map_native.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_native_truth_only",
                "exports": [{"topic": "/truth", "kind": "pose_truth"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    extracted_dir = output / "extracted"

    def fake_extract_native_rosbag_topic_map(
        *,
        bag_path,
        topic_map_json,
        output_dir,
        voxel_size_m,
        min_points,
    ):
        assert bag_path == bag
        assert topic_map_json == topic_map
        assert voxel_size_m == 0.75
        assert min_points == 3
        output_dir.mkdir(parents=True)
        (output_dir / "native_ros_extraction_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "fake-native-extraction",
                    "candidate_rows": 0,
                    "truth_rows": 1,
                }
            ),
            encoding="utf-8",
        )
        truth = TruthFrame(
            pd.DataFrame(
                {
                    "sequence_id": ["seq_native_truth_only"],
                    "time_s": [0.0],
                    "x_m": [0.0],
                    "y_m": [0.0],
                    "z_m": [10.0],
                }
            )
        )
        return SimpleNamespace(candidates=None, truth=truth, manifest={})

    monkeypatch.setattr(
        "raft_uav.mmuad.cli.extract_native_rosbag_topic_map",
        fake_extract_native_rosbag_topic_map,
    )

    with pytest.raises(SystemExit, match="no candidate rows"):
        mmuad_cli_main(
            [
                "--rosbag-path",
                str(bag),
                "--topic-map-file",
                str(topic_map),
                "--native-ros-extract-output-dir",
                str(extracted_dir),
                "--output-dir",
                str(output),
                "--submission-csv",
                str(output / "submission.csv"),
            ]
        )

    assert (extracted_dir / "native_ros_extraction_manifest.json").exists()
    assert not (output / "mmuad_estimates.csv").exists()


def test_native_ros_extraction_manifest_reports_missing_topic_map_topics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_missing_topics.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map_native.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_missing_topics",
                "exports": [
                    {
                        "topic": "/detector/pose",
                        "kind": "pose_candidate",
                        "source": "detector",
                    },
                    {"topic": "/ground_truth/pose", "kind": "pose_truth"},
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == []
            return []

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is None
    assert extracted.truth is None
    assert extracted.manifest["candidate_rows"] == 0
    assert extracted.manifest["truth_rows"] == 0
    assert extracted.manifest["extracted_messages"] == [
        {
            "topic": "/detector/pose",
            "kind": "pose_candidate",
            "status": "missing_topic",
        },
        {
            "topic": "/ground_truth/pose",
            "kind": "pose_truth",
            "status": "missing_topic",
        },
    ]
    saved = json.loads(
        (output / "native_ros_extraction_manifest.json").read_text(encoding="utf-8")
    )
    assert saved["extracted_messages"] == extracted.manifest["extracted_messages"]


def test_native_ros_extraction_manifest_reports_empty_matched_topics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_empty_matched_topics.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map_native.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_empty_matched_topics",
                "exports": [
                    {
                        "topic": "/camera/camera_info",
                        "kind": "camera_info_calibration",
                        "source": "cam0",
                    },
                    {
                        "topic": "/detector/pose",
                        "kind": "pose_candidate",
                        "source": "detector",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    camera_connection = SimpleNamespace(
        topic="/camera/camera_info",
        msgtype="sensor_msgs/msg/CameraInfo",
    )
    candidate_connection = SimpleNamespace(
        topic="/detector/pose",
        msgtype="geometry_msgs/msg/PoseStamped",
    )
    observed_calls: list[tuple[str, ...]] = []

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [camera_connection, candidate_connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            observed_calls.append(tuple(connection.topic for connection in connections))
            return []

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert observed_calls == [("/camera/camera_info",), ("/detector/pose",)]
    assert extracted.candidates is None
    assert extracted.truth is None
    assert extracted.manifest["candidate_rows"] == 0
    assert extracted.manifest["truth_rows"] == 0
    assert extracted.manifest["extracted_messages"] == [
        {
            "topic": "/camera/camera_info",
            "kind": "camera_info_calibration",
            "status": "matched_topic_no_messages",
            "msgtype": "sensor_msgs/msg/CameraInfo",
        },
        {
            "topic": "/detector/pose",
            "kind": "pose_candidate",
            "status": "matched_topic_no_messages",
            "msgtype": "geometry_msgs/msg/PoseStamped",
        },
    ]
    saved = json.loads(
        (output / "native_ros_extraction_manifest.json").read_text(encoding="utf-8")
    )
    assert saved["extracted_messages"] == extracted.manifest["extracted_messages"]


def test_native_ros_extraction_supports_image_timestamp_topics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_images.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map_native.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_images",
                "exports": [
                    {
                        "topic": "/camera/front/image_raw",
                        "kind": "image_timestamps",
                        "source": "front",
                    },
                    {
                        "topic": "/camera/front/image_compressed",
                        "kind": "compressed_image_timestamps",
                        "source": "front_compressed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    image_connection = SimpleNamespace(
        topic="/camera/front/image_raw",
        msgtype="sensor_msgs/msg/Image",
    )
    compressed_connection = SimpleNamespace(
        topic="/camera/front/image_compressed",
        msgtype="sensor_msgs/msg/CompressedImage",
    )
    image_message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=1, nanosec=250_000_000),
            frame_id="front_optical",
        ),
        height=480,
        width=640,
        encoding="rgb8",
        step=1920,
        data=b"rgb",
    )
    compressed_message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=2, nanosec=500_000_000),
            frame_id="front_optical",
        ),
        format="jpeg",
        data=b"jpeg-bytes",
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [image_connection, compressed_connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [image_connection, compressed_connection]
            return [
                (image_connection, 1_000_000_000, b"image"),
                (compressed_connection, 2_000_000_000, b"compressed"),
            ]

        def deserialize(self, rawdata, msgtype):
            if rawdata == b"image":
                assert msgtype == "sensor_msgs/msg/Image"
                return image_message
            if rawdata == b"compressed":
                assert msgtype == "sensor_msgs/msg/CompressedImage"
                return compressed_message
            raise AssertionError(f"unexpected rawdata: {rawdata!r}")

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is None
    assert extracted.truth is None
    assert extracted.image_timestamps is not None
    rows = extracted.image_timestamps
    assert rows["sequence_id"].tolist() == ["seq_images", "seq_images"]
    assert rows["time_s"].tolist() == [1.25, 2.5]
    assert rows["source"].tolist() == ["front", "front_compressed"]
    assert rows["frame_id"].tolist() == ["front_optical", "front_optical"]
    assert rows.loc[0, "height"] == 480
    assert rows.loc[0, "width"] == 640
    assert rows.loc[0, "encoding"] == "rgb8"
    assert rows.loc[1, "format"] == "jpeg"
    assert extracted.manifest["candidate_rows"] == 0
    assert extracted.manifest["truth_rows"] == 0
    assert extracted.manifest["image_timestamp_rows"] == 2
    assert [row["status"] for row in extracted.manifest["extracted_messages"]] == [
        "extracted",
        "extracted",
    ]

    saved_rows = pd.read_csv(output / "native_ros_image_timestamps.csv")
    saved_template = pd.read_csv(output / "native_ros_image_timestamp_template.csv")
    saved_manifest = json.loads(
        (output / "native_ros_extraction_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_rows["time_s"].tolist() == [1.25, 2.5]
    assert saved_template.columns.tolist() == ["sequence_id", "time_s", "x_m", "y_m", "z_m"]
    assert saved_template["time_s"].tolist() == [1.25, 2.5]
    assert saved_manifest["image_timestamp_rows"] == 2
    assert saved_manifest["image_timestamp_template_csv"] == str(
        output / "native_ros_image_timestamp_template.csv"
    )


def test_native_ros_extraction_supports_audio_timestamp_topics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_audio.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map_native.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_audio",
                "exports": [
                    {
                        "topic": "/microphone/audio",
                        "kind": "audio_timestamps",
                        "source": "mic",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    audio_connection = SimpleNamespace(
        topic="/microphone/audio",
        msgtype="audio_common_msgs/msg/AudioData",
    )
    audio_message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=3, nanosec=250_000_000),
            frame_id="microphone",
        ),
        sample_rate=48_000,
        channels=2,
        encoding="s16le",
        data=bytes(range(16)),
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [audio_connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [audio_connection]
            return [(audio_connection, 3_000_000_000, b"audio")]

        def deserialize(self, rawdata, msgtype):
            assert rawdata == b"audio"
            assert msgtype == "audio_common_msgs/msg/AudioData"
            return audio_message

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is None
    assert extracted.truth is None
    assert extracted.image_timestamps is None
    assert extracted.audio_timestamps is not None
    rows = extracted.audio_timestamps
    assert rows["sequence_id"].tolist() == ["seq_audio"]
    assert rows["time_s"].tolist() == [3.25]
    assert rows["source"].tolist() == ["mic"]
    assert rows["frame_id"].tolist() == ["microphone"]
    assert rows.loc[0, "sample_rate_hz"] == 48_000
    assert rows.loc[0, "channels"] == 2
    assert rows.loc[0, "data_length"] == 16
    assert rows.loc[0, "byte_count"] == 16
    assert rows.loc[0, "sample_count"] == 8
    assert rows.loc[0, "frame_count"] == 4
    assert rows.loc[0, "duration_s"] == pytest.approx(4 / 48_000)
    assert extracted.manifest["candidate_rows"] == 0
    assert extracted.manifest["truth_rows"] == 0
    assert extracted.manifest["image_timestamp_rows"] == 0
    assert extracted.manifest["audio_timestamp_rows"] == 1

    saved_rows = pd.read_csv(output / "native_ros_audio_timestamps.csv")
    saved_manifest = json.loads(
        (output / "native_ros_extraction_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_rows["time_s"].tolist() == [3.25]
    assert saved_manifest["audio_timestamp_rows"] == 1
    assert saved_manifest["audio_timestamps_csv"] == str(
        output / "native_ros_audio_timestamps.csv"
    )
    assert not (output / "native_ros_audio_timestamp_template.csv").exists()


def test_native_ros_extraction_supports_imu_timestamp_topics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_imu.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map_native.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_imu",
                "exports": [
                    {
                        "topic": "/os1_cloud_node1/imu",
                        "kind": "imu_timestamps",
                        "source": "os1_imu",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    imu_connection = SimpleNamespace(
        topic="/os1_cloud_node1/imu",
        msgtype="sensor_msgs/msg/Imu",
    )
    imu_message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=4, nanosec=500_000_000),
            frame_id="os1_imu",
        ),
        orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
        angular_velocity=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        linear_acceleration=SimpleNamespace(x=4.0, y=5.0, z=6.0),
        orientation_covariance=[0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03],
        angular_velocity_covariance=[0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3],
        linear_acceleration_covariance=[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [imu_connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [imu_connection]
            return [(imu_connection, 4_000_000_000, b"imu")]

        def deserialize(self, rawdata, msgtype):
            assert rawdata == b"imu"
            assert msgtype == "sensor_msgs/msg/Imu"
            return imu_message

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is None
    assert extracted.truth is None
    assert extracted.image_timestamps is None
    assert extracted.audio_timestamps is None
    assert extracted.imu_timestamps is not None
    rows = extracted.imu_timestamps
    assert rows["sequence_id"].tolist() == ["seq_imu"]
    assert rows["time_s"].tolist() == [4.5]
    assert rows["source"].tolist() == ["os1_imu"]
    assert rows["frame_id"].tolist() == ["os1_imu"]
    assert rows.loc[0, "orientation_w"] == pytest.approx(0.9)
    assert rows.loc[0, "angular_velocity_z_rad_s"] == pytest.approx(3.0)
    assert rows.loc[0, "linear_acceleration_y_m_s2"] == pytest.approx(5.0)
    assert rows.loc[0, "orientation_covariance_zz"] == pytest.approx(0.03)
    assert rows.loc[0, "angular_velocity_covariance_yy"] == pytest.approx(0.2)
    assert rows.loc[0, "linear_acceleration_covariance_xx"] == pytest.approx(1.0)
    assert extracted.manifest["candidate_rows"] == 0
    assert extracted.manifest["truth_rows"] == 0
    assert extracted.manifest["image_timestamp_rows"] == 0
    assert extracted.manifest["audio_timestamp_rows"] == 0
    assert extracted.manifest["imu_timestamp_rows"] == 1

    saved_rows = pd.read_csv(output / "native_ros_imu_timestamps.csv")
    saved_manifest = json.loads(
        (output / "native_ros_extraction_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_rows["time_s"].tolist() == [4.5]
    assert saved_manifest["imu_timestamp_rows"] == 1
    assert saved_manifest["imu_timestamps_csv"] == str(
        output / "native_ros_imu_timestamps.csv"
    )
    assert not (output / "native_ros_imu_timestamp_template.csv").exists()


def test_native_ros_extraction_supports_bounding_box3d_topics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_boxes.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map_native.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_boxes",
                "exports": [
                    {
                        "topic": "/detector/boxes",
                        "kind": "bounding_box3d_array_candidate",
                        "source": "boxes",
                    },
                    {
                        "topic": "/ground_truth/box",
                        "kind": "bounding_box3d_truth",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate_connection = SimpleNamespace(
        topic="/detector/boxes",
        msgtype="vision_msgs/msg/BoundingBox3DArray",
    )
    truth_connection = SimpleNamespace(
        topic="/ground_truth/box",
        msgtype="vision_msgs/msg/BoundingBox3D",
    )
    candidate_message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=3, nanosec=0),
            frame_id="world",
        ),
        boxes=[
            SimpleNamespace(
                id="box-a",
                center=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0, z=3.0)
                ),
                size=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                label="quadrotor",
                confidence=0.9,
            )
        ],
    )
    truth_message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=3, nanosec=100_000_000),
            frame_id="world",
        ),
        center=SimpleNamespace(position=SimpleNamespace(x=1.5, y=2.5, z=3.5)),
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [candidate_connection, truth_connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [candidate_connection, truth_connection]
            return [
                (candidate_connection, 3_000_000_000, b"candidate-boxes"),
                (truth_connection, 3_100_000_000, b"truth-box"),
            ]

        def deserialize(self, rawdata, msgtype):
            if rawdata == b"candidate-boxes":
                assert msgtype == "vision_msgs/msg/BoundingBox3DArray"
                return candidate_message
            if rawdata == b"truth-box":
                assert msgtype == "vision_msgs/msg/BoundingBox3D"
                return truth_message
            raise AssertionError(f"unexpected rawdata: {rawdata!r}")

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is not None
    assert extracted.truth is not None
    candidates = extracted.candidates.rows
    truth = extracted.truth.rows
    assert len(candidates) == 1
    assert len(truth) == 1
    assert candidates.loc[0, "sequence_id"] == "seq_boxes"
    assert candidates.loc[0, "source"] == "boxes"
    assert candidates.loc[0, "track_id"] == "box-a"
    assert candidates.loc[0, "x_m"] == 1.0
    assert candidates.loc[0, "confidence"] == 0.9
    assert candidates.loc[0, "class_name"] == "quadrotor"
    assert truth.loc[0, "time_s"] == 3.1
    assert truth.loc[0, "x_m"] == 1.5
    assert extracted.manifest["candidate_rows"] == 1
    assert extracted.manifest["truth_rows"] == 1
    assert [row["kind"] for row in extracted.manifest["extracted_messages"]] == [
        "bounding_box3d_array_candidate",
        "bounding_box3d_truth",
    ]
    assert (output / "native_ros_candidates.csv").exists()
    assert (output / "native_ros_truth.csv").exists()


def test_topic_map_exports_load_json_candidates_and_truth(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "radar.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"stamp": 0.0, "px": 0.0, "py": 0.0, "pz": 5.0},
                    {"stamp": 1.0, "px": 1.0, "py": 0.0, "pz": 5.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    (exports / "truth.json").write_text(
        json.dumps(
            {
                "truth": [
                    {"stamp": 0.0, "px": 0.0, "py": 0.0, "pz": 5.0},
                    {"stamp": 1.0, "px": 1.0, "py": 0.0, "pz": 5.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    topic_map = tmp_path / "topic_map_json.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_ros_json",
                "exports": [
                    {
                        "kind": "candidate",
                        "path": "radar.json",
                        "source": "radar",
                        "column_aliases": {
                            "stamp": "time_s",
                            "px": "x_m",
                            "py": "y_m",
                            "pz": "z_m",
                        },
                    },
                    {
                        "kind": "truth",
                        "path": "truth.json",
                        "column_aliases": {
                            "stamp": "time_s",
                            "px": "x_m",
                            "py": "y_m",
                            "pz": "z_m",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert bundle.candidates.rows["sequence_id"].tolist() == [
        "seq_ros_json",
        "seq_ros_json",
    ]
    assert bundle.candidates.rows["source"].tolist() == ["radar", "radar"]
    assert bundle.truth is not None
    assert bundle.truth.rows["time_s"].tolist() == [0.0, 1.0]
    assert [row["rows"] for row in bundle.manifest["loaded_exports"]] == [2, 2]


def test_topic_map_exports_load_ros_shaped_json_pose_rows(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "radar_pose.json").write_text(
        json.dumps(
            {
                "poses": [
                    {
                        "header": {
                            "stamp": {"sec": 4, "nanosec": 125_000_000},
                            "frame_id": "radar_frame",
                        },
                        "child_frame_id": "track_a",
                        "pose": {
                            "position": {"x": 1.0, "y": 2.0, "z": 7.0}
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (exports / "truth_pose.json").write_text(
        json.dumps(
            {
                "header": {"stamp": {"sec": 4, "nanosec": 125_000_000}},
                "poses": [
                    {
                        "pose": {
                            "position": {"x": 1.0, "y": 2.0, "z": 7.0}
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    topic_map = tmp_path / "topic_map_pose_json.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_ros_pose_json",
                "exports": [
                    {
                        "kind": "pose_candidate",
                        "path": "radar_pose.json",
                        "source": "radar_pose",
                    },
                    {"kind": "pose_truth", "path": "truth_pose.json"},
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    row = bundle.candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_ros_pose_json"
    assert abs(float(row["time_s"]) - 4.125) < 1e-12
    assert row["source"] == "radar_pose"
    assert row["track_id"] == "track_a"
    assert row[["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 7.0]
    assert bundle.truth is not None
    assert bundle.truth.rows["time_s"].tolist() == [4.125]


def test_topic_map_exports_load_numpy_trajectory_files(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    trajectory = np.array(
        [
            [0.0, 0.0, 0.0, 5.0],
            [1.0, 1.0, 0.0, 5.0],
        ]
    )
    np.save(exports / "radar_trajectory.npy", trajectory)
    np.save(exports / "truth.npy", trajectory)
    topic_map = tmp_path / "topic_map_numpy.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_ros_numpy",
                "exports": [
                    {
                        "kind": "candidate",
                        "path": "radar_trajectory.npy",
                        "source": "radar",
                        "track_id": "radar-track",
                        "class_name": "Mavic3",
                    },
                    {
                        "kind": "truth",
                        "path": "truth.npy",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert bundle.candidates.rows["sequence_id"].tolist() == [
        "seq_ros_numpy",
        "seq_ros_numpy",
    ]
    assert bundle.candidates.rows["source"].tolist() == ["radar", "radar"]
    assert bundle.candidates.rows["track_id"].tolist() == ["radar-track", "radar-track"]
    assert bundle.candidates.rows["class_name"].tolist() == ["Mavic3", "Mavic3"]
    assert bundle.truth is not None
    assert bundle.truth.rows["time_s"].tolist() == [0.0, 1.0]
    loaded = bundle.manifest["loaded_exports"]
    assert [row["rows"] for row in loaded] == [2, 2]


def test_topic_map_exports_treat_native_truth_kinds_as_truth(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [5.0, 5.0],
        }
    ).to_csv(exports / "truth_pose.csv", index=False)
    pd.DataFrame(
        {
            "stamp": [0.0, 1.0],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [5.0, 5.0],
        }
    ).to_csv(exports / "detections.csv", index=False)
    topic_map = tmp_path / "topic_map_native_kinds.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_native_kind_exports",
                "exports": [
                    {
                        "kind": "pose_truth",
                        "path": "truth_pose.csv",
                        "column_aliases": {
                            "stamp": "time_s",
                            "x": "x_m",
                            "y": "y_m",
                            "z": "z_m",
                        },
                    },
                    {
                        "kind": "odometry_candidate",
                        "path": "detections.csv",
                        "source": "odom_detector",
                        "column_aliases": {
                            "stamp": "time_s",
                            "x": "x_m",
                            "y": "y_m",
                            "z": "z_m",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert bundle.truth is not None
    assert bundle.truth.rows["sequence_id"].tolist() == [
        "seq_native_kind_exports",
        "seq_native_kind_exports",
    ]
    assert bundle.candidates.rows["source"].tolist() == [
        "odom_detector",
        "odom_detector",
    ]


def test_topic_map_exports_convert_radar_polar_candidate_tables(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    pd.DataFrame(
        {
            "stamp": [1.25],
            "rng": [10.0],
            "bearing": [90.0],
            "track": ["r1"],
        }
    ).to_csv(exports / "radar_polar.csv", index=False)
    topic_map = tmp_path / "topic_map_radar_polar.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_radar_polar_topic",
                "exports": [
                    {
                        "kind": "radar_polar_candidate",
                        "path": "radar_polar.csv",
                        "source": "radar0",
                        "azimuth_convention": "north-clockwise",
                        "angle_unit": "deg",
                        "range_std_m": 3.0,
                        "z_std_m": 4.0,
                        "column_aliases": {
                            "stamp": "time_s",
                            "rng": "range_m",
                            "bearing": "azimuth_deg",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert bundle.candidates.rows["sequence_id"].tolist() == ["seq_radar_polar_topic"]
    row = bundle.candidates.rows.iloc[0]
    assert row["source"] == "radar0"
    assert row["track_id"] == "r1"
    assert abs(float(row["time_s"]) - 1.25) < 1.0e-12
    assert abs(float(row["x_m"]) - 10.0) < 1.0e-9
    assert abs(float(row["y_m"])) < 1.0e-9
    assert float(row["std_xy_m"]) == 3.0
    assert float(row["std_z_m"]) == 4.0
    assert [entry["rows"] for entry in bundle.manifest["loaded_exports"]] == [1]


def test_topic_map_exports_convert_detection3d_flattened_tables(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    pd.DataFrame(
        {
            "header.stamp.sec": [5],
            "header.stamp.nanosec": [750_000_000],
            "id": ["det-7"],
            "bbox.center.position.x": [1.0],
            "bbox.center.position.y": [2.0],
            "bbox.center.position.z": [3.0],
            "results.0.hypothesis.class_id": ["Mavic3"],
            "results.0.hypothesis.score": [0.7],
        }
    ).to_csv(exports / "detections3d.csv", index=False)
    topic_map = tmp_path / "topic_map_detection3d.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_detection3d_topic",
                "exports": [
                    {
                        "kind": "detection3d_array_candidate",
                        "path": "detections3d.csv",
                        "source": "detector3d",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert bundle.candidates.rows["sequence_id"].tolist() == ["seq_detection3d_topic"]
    row = bundle.candidates.rows.iloc[0]
    assert row["source"] == "detector3d"
    assert row["track_id"] == "det-7"
    assert abs(float(row["time_s"]) - 5.75) < 1.0e-12
    assert row[["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]
    assert float(row["confidence"]) == 0.7
    assert row["class_name"] == "Mavic3"
    assert [entry["rows"] for entry in bundle.manifest["loaded_exports"]] == [1]


def test_topic_map_exports_convert_camera_detection_tables(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    camera = exports / "cam0"
    camera.mkdir(parents=True)
    pd.DataFrame(
        {
            "stamp": [2.5],
            "bbox": ["[40, 40, 20, 20]"],
            "depth_m": [5.0],
            "score": [0.75],
        }
    ).to_csv(camera / "detections.csv", index=False)
    (camera / "camera_info.json").write_text(
        json.dumps(
            {
                "width": 100,
                "height": 100,
                "k": [
                    100.0,
                    0.0,
                    50.0,
                    0.0,
                    100.0,
                    50.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            }
        ),
        encoding="utf-8",
    )
    topic_map = tmp_path / "topic_map_camera.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_camera_topic",
                "exports": [
                    {
                        "kind": "camera_detections_candidate",
                        "path": "cam0/detections.csv",
                        "source": "cam0",
                        "column_aliases": {"stamp": "time_s"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert bundle.candidates.rows["sequence_id"].tolist() == ["seq_camera_topic"]
    row = bundle.candidates.rows.iloc[0]
    assert row["source"] == "cam0"
    assert abs(float(row["time_s"]) - 2.5) < 1.0e-12
    assert (row["x_m"], row["y_m"], row["z_m"]) == (0.0, 0.0, 5.0)
    assert float(row["confidence"]) == 0.75
    assert [entry["rows"] for entry in bundle.manifest["loaded_exports"]] == [1]


def test_topic_map_exports_convert_detection2d_candidate_tables(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    camera = exports / "camera_front"
    camera.mkdir(parents=True)
    pd.DataFrame(
        {
            "stamp": [3.5],
            "center_x": [50.0],
            "center_y": [50.0],
            "depth": [6.0],
            "score": [0.6],
        }
    ).to_csv(camera / "detections2d.csv", index=False)
    (exports / "camera_info.json").write_text(
        json.dumps(
            {
                "source": "camera_front",
                "fx": 100.0,
                "fy": 100.0,
                "cx": 50.0,
                "cy": 50.0,
            }
        ),
        encoding="utf-8",
    )
    topic_map = tmp_path / "topic_map_detection2d.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_detection2d_topic",
                "exports": [
                    {
                        "kind": "detection2d_array_candidate",
                        "path": "camera_front/detections2d.csv",
                        "source": "camera_front",
                        "camera_calibration_file": "camera_info.json",
                        "column_aliases": {
                            "stamp": "time_s",
                            "score": "confidence",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert bundle.candidates.rows["sequence_id"].tolist() == ["seq_detection2d_topic"]
    row = bundle.candidates.rows.iloc[0]
    assert row["source"] == "camera_front"
    assert abs(float(row["time_s"]) - 3.5) < 1.0e-12
    assert (row["x_m"], row["y_m"], row["z_m"]) == (0.0, 0.0, 6.0)
    assert float(row["confidence"]) == 0.6


def test_topic_map_exports_cluster_pointcloud2_candidate_tables(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    pd.DataFrame(
        {
            "stamp": [7.5, 7.5, 7.5],
            "x": [0.0, 0.1, 0.2],
            "y": [0.0, 0.0, 0.1],
            "z": [5.0, 5.1, 5.0],
        }
    ).to_csv(exports / "lidar_points.csv", index=False)
    topic_map = tmp_path / "topic_map_pointcloud2.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_pointcloud2_exports",
                "exports": [
                    {
                        "kind": "pointcloud2_candidate",
                        "path": "lidar_points.csv",
                        "source": "lidar",
                        "min_cluster_points": 3,
                        "voxel_size_m": 0.5,
                        "column_aliases": {
                            "stamp": "time_s",
                            "x": "x_m",
                            "y": "y_m",
                            "z": "z_m",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert len(bundle.candidates.rows) == 1
    row = bundle.candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_pointcloud2_exports"
    assert row["source"] == "lidar"
    assert abs(float(row["time_s"]) - 7.5) < 1.0e-9


def test_topic_map_exports_cluster_pointcloud2_json_point_rows(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "lidar_points.json").write_text(
        json.dumps(
            {
                "points": [
                    {"stamp": 7.5, "x": 0.0, "y": 0.0, "z": 5.0},
                    {"stamp": 7.5, "x": 0.1, "y": 0.0, "z": 5.1},
                    {"stamp": 7.5, "x": 0.2, "y": 0.1, "z": 5.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    topic_map = tmp_path / "topic_map_pointcloud2_json.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_pointcloud2_json_exports",
                "exports": [
                    {
                        "kind": "pointcloud2_candidate",
                        "path": "lidar_points.json",
                        "source": "lidar",
                        "min_cluster_points": 3,
                        "voxel_size_m": 0.5,
                        "column_aliases": {
                            "stamp": "time_s",
                            "x": "x_m",
                            "y": "y_m",
                            "z": "z_m",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    assert len(bundle.candidates.rows) == 1
    row = bundle.candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_pointcloud2_json_exports"
    assert row["source"] == "lidar"
    assert abs(float(row["time_s"]) - 7.5) < 1.0e-9


def test_topic_map_exports_project_geodetic_candidate_and_truth_tables(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    pd.DataFrame(
        {
            "timestamp_s": [2.5],
            "latitude": [35.0],
            "longitude": [-78.0],
            "altitude": [105.0],
        }
    ).to_csv(exports / "gps_fix.csv", index=False)
    (exports / "truth_geopose.json").write_text(
        json.dumps(
            {
                "poses": [
                    {
                        "time_s": 2.5,
                        "lat": 35.0,
                        "lon": -78.0,
                        "alt": 107.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    topic_map = tmp_path / "topic_map_geodetic.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_geodetic_exports",
                "exports": [
                    {
                        "kind": "navsatfix_candidate",
                        "path": "gps_fix.csv",
                        "source": "gps",
                        "enu_origin_lla": [35.0, -78.0, 100.0],
                    },
                    {
                        "kind": "geopose_truth",
                        "path": "truth_geopose.json",
                        "enu_origin_lla": "35.0,-78.0,100.0",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    candidate_row = bundle.candidates.rows.iloc[0]
    assert candidate_row["sequence_id"] == "seq_geodetic_exports"
    assert candidate_row["source"] == "gps"
    assert abs(float(candidate_row["x_m"])) < 1.0e-9
    assert abs(float(candidate_row["y_m"])) < 1.0e-9
    assert abs(float(candidate_row["z_m"]) - 5.0) < 1.0e-9
    assert bundle.truth is not None
    truth_row = bundle.truth.rows.iloc[0]
    assert abs(float(truth_row["z_m"]) - 7.0) < 1.0e-9


def test_topic_map_exports_project_geodetic_json_fix_wrappers(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "gps_fixes.json").write_text(
        json.dumps(
            {
                "fixes": [
                    {
                        "timestamp_s": 3.0,
                        "latitude": 35.0,
                        "longitude": -78.0,
                        "altitude": 101.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    topic_map = tmp_path / "topic_map_geodetic_fixes.json"
    topic_map.write_text(
        json.dumps(
            {
                "sequence_id": "seq_geodetic_fixes",
                "exports": [
                    {
                        "kind": "navsatfix_candidate",
                        "path": "gps_fixes.json",
                        "source": "gps",
                        "enu_origin_lla": [35.0, -78.0, 100.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = load_topic_map_exports(topic_map, base_dir=exports)

    row = bundle.candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_geodetic_fixes"
    assert row["source"] == "gps"
    assert abs(float(row["time_s"]) - 3.0) < 1.0e-9
    assert abs(float(row["x_m"])) < 1.0e-9
    assert abs(float(row["y_m"])) < 1.0e-9
    assert abs(float(row["z_m"]) - 1.5) < 1.0e-9


def test_ros2_metadata_inspection_and_topic_map_template(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  storage_identifier: sqlite3",
                "  serialization_format: cdr",
                "  duration:",
                "    nanoseconds: 2500000000",
                "  starting_time:",
                "    nanoseconds_since_epoch: 1700000000123456789",
                "  message_count: 9",
                "  relative_file_paths:",
                "    - bag_0.db3",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /radar/points",
                "        type: sensor_msgs/msg/PointCloud2",
                "        serialization_format: cdr",
                "      message_count: 3",
                "    - topic_metadata:",
                "        name: /ground_truth",
                "        type: geometry_msgs/msg/PoseStamped",
                "        serialization_format: cdr",
                "      message_count: 3",
                "    - topic_metadata:",
                "        name: /detector/odom",
                "        type: nav_msgs/msg/Odometry",
                "        serialization_format: cdr",
                "      message_count: 3",
            ]
        ),
        encoding="utf-8",
    )
    (bag / "bag_0.db3").write_bytes(b"")
    report = inspect_rosbag(bag)
    assert report["kind"] == "ros2_bag_directory"
    assert report["storage_identifier"] == "sqlite3"
    assert report["serialization_format"] == "cdr"
    assert report["relative_file_paths"] == ["bag_0.db3"]
    assert report["db3_files"] == ["bag_0.db3"]
    assert report["total_message_count"] == 9
    assert report["duration_s"] == 2.5
    assert abs(report["starting_time_s"] - 1700000000.1234567) < 1.0e-6
    assert len(report["topics"]) == 3
    assert report["topics"][0]["serialization_format"] == "cdr"
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))
    assert payload["schema"] == "raft-uav-mmuad-topic-map-v1"
    assert payload["template_mode"] == "export"
    assert [entry["kind"] for entry in payload["exports"]] == [
        "pointcloud2_candidate",
        "pose_truth",
        "odometry_candidate",
    ]
    assert all("path" in entry for entry in payload["exports"])
    assert payload["exports"][0]["source"] == "radar_points"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_native_mode_is_extraction_ready(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_native_template"
    seq.mkdir()
    (seq / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  relative_file_paths:",
                "    - data_0.db3",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /livox/points",
                "        type: sensor_msgs/msg/PointCloud2",
                "      message_count: 3",
                "    - topic_metadata:",
                "        name: /mmwave/range_azimuth",
                "        type: custom_msgs/msg/RadarPolarArray",
                "      message_count: 3",
                "    - topic_metadata:",
                "        name: /camera/detections",
                "        type: vision_msgs/msg/Detection2DArray",
                "      message_count: 3",
                "    - topic_metadata:",
                "        name: /ground_truth/pose",
                "        type: geometry_msgs/msg/PoseStamped",
                "      message_count: 3",
            ]
        ),
        encoding="utf-8",
    )
    (seq / "data_0.db3").write_bytes(b"fake sqlite bag")

    report = inspect_rosbag(seq)
    template = write_topic_map_template(
        report,
        seq / "topic_map_native.json",
        template_mode="native",
    )
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert payload["template_mode"] == "native"
    assert "Native ROS extraction template" in payload["description"]
    assert [entry["kind"] for entry in payload["exports"]] == [
        "pointcloud2_candidate",
        "radar_polar_candidate",
        "camera_detections_candidate",
        "pose_truth",
    ]
    assert all("path" not in entry for entry in payload["exports"])
    assert all("column_aliases" not in entry for entry in payload["exports"])
    radar_export = payload["exports"][1]
    assert radar_export["angle_unit"] == "rad"
    camera_export = payload["exports"][2]
    assert camera_export["camera_calibration_file"] == "PATH/TO/camera_info.json"
    assert camera_export["camera_fixed_depth_m"] == "SET_DEPTH_OR_REMOVE_IF_MESSAGE_HAS_DEPTH"

    discovered = discover_sequence_paths(tmp_path)
    assert len(discovered) == 1
    assert discovered[0].native_topic_map_jsons == (template,)
    assert discovered[0].rosbag_paths == (seq,)


def test_cli_writes_native_topic_map_template(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/point",
                "        type: geometry_msgs/msg/PointStamped",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    template = output / "topic_map_native.json"

    status = mmuad_cli_main(
        [
            "--rosbag-path",
            str(bag),
            "--topic-map-template-json",
            str(template),
            "--topic-map-template-mode",
            "native",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(template.read_text(encoding="utf-8"))
    assert payload["template_mode"] == "native"
    assert payload["exports"][0]["kind"] == "point_candidate"
    assert "path" not in payload["exports"][0]


def test_standalone_mcap_inspection_reports_missing_native_reader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq001.mcap"
    mcap.write_bytes(b"not-a-real-mcap")
    monkeypatch.setitem(sys.modules, "rosbags.highlevel", None)

    report = inspect_rosbag(mcap)

    assert report["kind"] == "ros2_recording_file"
    assert report["suffix"] == ".mcap"
    assert report["storage_identifier"] == "mcap"
    assert report["rosbags_available"] is False
    assert report["topics"] == []
    assert "rosbags" in report["recommendation"]


def test_standalone_mcap_inspection_uses_rosbags_topics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq001.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [
                SimpleNamespace(
                    topic="/livox/points",
                    msgtype="sensor_msgs/msg/PointCloud2",
                    msgcount=4,
                    ext=SimpleNamespace(serialization_format="cdr"),
                ),
                SimpleNamespace(
                    topic="/ground_truth/pose",
                    msgtype="geometry_msgs/msg/PoseStamped",
                    msgcount=4,
                    ext=SimpleNamespace(serialization_format="cdr"),
                ),
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    report = inspect_rosbag(mcap)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert report["kind"] == "ros2_recording_file"
    assert report["rosbags_available"] is True
    assert report["total_message_count"] == 8
    assert report["topics"] == [
        {
            "name": "/livox/points",
            "type": "sensor_msgs/msg/PointCloud2",
            "message_count": 4,
            "serialization_format": "cdr",
        },
        {
            "name": "/ground_truth/pose",
            "type": "geometry_msgs/msg/PoseStamped",
            "message_count": 4,
            "serialization_format": "cdr",
        },
    ]
    assert [entry["kind"] for entry in payload["exports"]] == [
        "pointcloud2_candidate",
        "pose_truth",
    ]
    assert payload["exports"][0]["source"] == "livox_points"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_livox_custom_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /livox/lidar",
                "        type: livox_ros_driver2/msg/CustomMsg",
                "      message_count: 4",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    export = payload["exports"][0]
    assert export["kind"] == "livox_custommsg_candidate"
    assert export["source"] == "livox_lidar"


def test_native_ros_extraction_supports_detection2d_camera_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_camera.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map.json"
    camera_info = tmp_path / "camera_info.json"
    output = tmp_path / "native_out"
    camera_info.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 40.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_camera",
                "exports": [
                    {
                        "topic": "/camera/detections",
                        "kind": "camera_detections_candidate",
                        "source": "cam0",
                        "camera_calibration_file": "camera_info.json",
                        "camera_fixed_depth_m": 10.0,
                        "std_xy_m": 1.5,
                        "std_z_m": 3.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    connection = SimpleNamespace(
        topic="/camera/detections",
        msgtype="vision_msgs/msg/Detection2DArray",
    )
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=2, nanosec=0),
            frame_id="cam0",
        ),
        detections=[
            SimpleNamespace(
                id="det-a",
                bbox=SimpleNamespace(
                    center=SimpleNamespace(x=60.0, y=50.0),
                    size_x=8.0,
                    size_y=4.0,
                ),
                results=[
                    SimpleNamespace(
                        hypothesis=SimpleNamespace(class_id="quad", score=0.75)
                    )
                ],
            )
        ],
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [connection]
            return [(connection, 2_000_000_000, b"raw")]

        def deserialize(self, rawdata, msgtype):
            assert rawdata == b"raw"
            assert msgtype == "vision_msgs/msg/Detection2DArray"
            return message

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is not None
    rows = extracted.candidates.rows
    assert len(rows) == 1
    assert rows.loc[0, "sequence_id"] == "seq_camera"
    assert rows.loc[0, "source"] == "cam0"
    assert rows.loc[0, "track_id"] == "det-a"
    assert rows.loc[0, "x_m"] == 1.0
    assert rows.loc[0, "y_m"] == 1.0
    assert rows.loc[0, "z_m"] == 10.0
    assert rows.loc[0, "std_xy_m"] == 1.5
    assert rows.loc[0, "std_z_m"] == 3.0
    assert rows.loc[0, "confidence"] == 0.75
    assert rows.loc[0, "class_name"] == "quad"
    assert extracted.manifest["candidate_rows"] == 1
    assert extracted.manifest["extracted_messages"][0]["status"] == "extracted"
    assert (output / "native_ros_candidates.csv").exists()


def test_native_ros_extraction_accepts_singular_detection2d_template_kind(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_camera_singular.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map.json"
    camera_info = tmp_path / "camera_info.json"
    camera_info.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 40.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_camera_singular",
                "exports": [
                    {
                        "topic": "/camera/best_detection",
                        "kind": "camera_detection_candidate",
                        "source": "cam0",
                        "camera_calibration_file": "camera_info.json",
                        "camera_fixed_depth_m": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    connection = SimpleNamespace(
        topic="/camera/best_detection",
        msgtype="vision_msgs/msg/Detection2D",
    )
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=3, nanosec=0),
            frame_id="cam0",
        ),
        id="det-b",
        bbox=SimpleNamespace(
            center=SimpleNamespace(x=70.0, y=40.0),
            size_x=8.0,
            size_y=4.0,
        ),
        results=[
            SimpleNamespace(hypothesis=SimpleNamespace(class_id="fixed-wing", score=0.8))
        ],
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [connection]
            return [(connection, 3_000_000_000, b"raw")]

        def deserialize(self, rawdata, msgtype):
            assert rawdata == b"raw"
            assert msgtype == "vision_msgs/msg/Detection2D"
            return message

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
    )

    assert extracted.candidates is not None
    rows = extracted.candidates.rows
    assert len(rows) == 1
    assert rows.loc[0, "sequence_id"] == "seq_camera_singular"
    assert rows.loc[0, "source"] == "cam0"
    assert rows.loc[0, "track_id"] == "det-b"
    assert rows.loc[0, "x_m"] == 2.0
    assert rows.loc[0, "y_m"] == 0.0
    assert rows.loc[0, "z_m"] == 10.0
    assert rows.loc[0, "confidence"] == 0.8
    assert rows.loc[0, "class_name"] == "fixed-wing"
    assert extracted.manifest["candidate_rows"] == 1
    assert extracted.manifest["extracted_messages"][0]["kind"] == "camera_detection_candidate"
    assert extracted.manifest["extracted_messages"][0]["status"] == "extracted"


def test_native_ros_extraction_supports_plain_point_topics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_point.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map.json"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_point",
                "exports": [
                    {
                        "topic": "/detector/point",
                        "kind": "point_candidate",
                        "source": "detector_point",
                        "std_xy_m": 1.0,
                        "std_z_m": 2.0,
                        "confidence": 0.7,
                    },
                    {
                        "topic": "/ground_truth/point",
                        "kind": "point_truth",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate_connection = SimpleNamespace(
        topic="/detector/point",
        msgtype="geometry_msgs/msg/Point",
    )
    truth_connection = SimpleNamespace(
        topic="/ground_truth/point",
        msgtype="geometry_msgs/msg/Point",
    )
    messages = {
        candidate_connection.topic: SimpleNamespace(x=1.0, y=2.0, z=3.0),
        truth_connection.topic: SimpleNamespace(x=1.5, y=2.5, z=3.5),
    }

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [candidate_connection, truth_connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [candidate_connection, truth_connection]
            return [
                (candidate_connection, 4_000_000_000, b"candidate"),
                (truth_connection, 4_000_000_000, b"truth"),
            ]

        def deserialize(self, rawdata, msgtype):
            if rawdata == b"candidate":
                assert msgtype == "geometry_msgs/msg/Point"
                return messages[candidate_connection.topic]
            if rawdata == b"truth":
                assert msgtype == "geometry_msgs/msg/Point"
                return messages[truth_connection.topic]
            raise AssertionError(f"unexpected rawdata: {rawdata!r}")

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
    )

    assert extracted.candidates is not None
    assert extracted.truth is not None
    candidate = extracted.candidates.rows.iloc[0]
    truth = extracted.truth.rows.iloc[0]
    assert candidate["sequence_id"] == "seq_point"
    assert candidate["source"] == "detector_point"
    assert candidate[["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]
    assert candidate["std_xy_m"] == 1.0
    assert candidate["std_z_m"] == 2.0
    assert candidate["confidence"] == 0.7
    assert truth[["x_m", "y_m", "z_m"]].tolist() == [1.5, 2.5, 3.5]
    assert extracted.manifest["candidate_rows"] == 1
    assert extracted.manifest["truth_rows"] == 1


def test_native_ros_extraction_supports_tracked_object_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_tracked_objects.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_tracked_objects",
                "exports": [
                    {
                        "topic": "/tracker/objects",
                        "kind": "tracked_objects_candidate",
                        "source": "tracker_objects",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    connection = SimpleNamespace(
        topic="/tracker/objects",
        msgtype="autoware_auto_perception_msgs/msg/TrackedObjects",
    )
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=5, nanosec=500_000_000),
            frame_id="map",
        ),
        objects=[
            SimpleNamespace(
                track_id="track-42",
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=4.0, y=5.0, z=6.0)
                ),
                label="uav",
                confidence=0.88,
            )
        ],
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [connection]
            return [(connection, 5_500_000_000, b"tracked-objects")]

        def deserialize(self, rawdata, msgtype):
            assert rawdata == b"tracked-objects"
            assert msgtype == "autoware_auto_perception_msgs/msg/TrackedObjects"
            return message

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is not None
    rows = extracted.candidates.rows
    assert len(rows) == 1
    assert rows.loc[0, "sequence_id"] == "seq_tracked_objects"
    assert rows.loc[0, "source"] == "tracker_objects"
    assert rows.loc[0, "track_id"] == "track-42"
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [4.0, 5.0, 6.0]
    assert rows.loc[0, "class_name"] == "uav"
    assert rows.loc[0, "confidence"] == 0.88
    assert extracted.manifest["candidate_rows"] == 1
    assert extracted.manifest["extracted_messages"][0]["kind"] == (
        "tracked_objects_candidate"
    )
    assert (output / "native_ros_candidates.csv").exists()


def test_native_ros_extraction_uses_camera_info_for_detection2d_intrinsics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_camera_info.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_camera_info",
                "exports": [
                    {
                        "topic": "/camera/camera_info",
                        "kind": "camera_info_calibration",
                        "source": "cam0",
                        "translation_m": [1.0, 2.0, 3.0],
                    },
                    {
                        "topic": "/camera/detections",
                        "kind": "camera_detections_candidate",
                        "source": "cam0",
                        "camera_fixed_depth_m": 10.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    camera_info_connection = SimpleNamespace(
        topic="/camera/camera_info",
        msgtype="sensor_msgs/msg/CameraInfo",
    )
    detection_connection = SimpleNamespace(
        topic="/camera/detections",
        msgtype="vision_msgs/msg/Detection2DArray",
    )
    camera_info_message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=1, nanosec=0),
            frame_id="cam0",
        ),
        k=[100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0],
    )
    detection_message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=2, nanosec=0),
            frame_id="cam0",
        ),
        detections=[
            SimpleNamespace(
                id="det-from-camera-info",
                bbox=SimpleNamespace(
                    center=SimpleNamespace(x=60.0, y=50.0),
                    size_x=8.0,
                    size_y=4.0,
                ),
                results=[
                    SimpleNamespace(
                        hypothesis=SimpleNamespace(class_id="quad", score=0.8)
                    )
                ],
            )
        ],
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [detection_connection, camera_info_connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            if connections == [camera_info_connection]:
                return [(camera_info_connection, 1_000_000_000, b"camera-info")]
            if connections == [detection_connection]:
                return [(detection_connection, 2_000_000_000, b"detections")]
            raise AssertionError(f"unexpected native ROS connections: {connections!r}")

        def deserialize(self, rawdata, msgtype):
            if rawdata == b"camera-info":
                assert msgtype == "sensor_msgs/msg/CameraInfo"
                return camera_info_message
            if rawdata == b"detections":
                assert msgtype == "vision_msgs/msg/Detection2DArray"
                return detection_message
            raise AssertionError(f"unexpected rawdata: {rawdata!r}")

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is not None
    rows = extracted.candidates.rows
    assert len(rows) == 1
    assert rows.loc[0, "sequence_id"] == "seq_camera_info"
    assert rows.loc[0, "source"] == "cam0"
    assert rows.loc[0, "track_id"] == "det-from-camera-info"
    assert rows.loc[0, "x_m"] == 2.0
    assert rows.loc[0, "y_m"] == 3.0
    assert rows.loc[0, "z_m"] == 13.0
    assert rows.loc[0, "confidence"] == 0.8
    assert extracted.manifest["extracted_messages"][0]["kind"] == "camera_info_calibration"
    assert extracted.manifest["extracted_messages"][0]["source"] == "cam0"
    assert extracted.manifest["extracted_messages"][1]["kind"] == "camera_detections_candidate"
    assert (output / "native_ros_candidates.csv").exists()


def test_native_ros_livox_custom_message_to_points_accepts_custom_points() -> None:
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=5, nanosec=100_000_000)),
        points=[
            SimpleNamespace(
                x=1.0,
                y=2.0,
                z=3.0,
                reflectivity=42,
                offset_time=10_000,
                tag=7,
                line=2,
            )
        ],
    )

    rows = livox_custom_message_to_points(
        message,
        sequence_id="seq_livox",
        time_s=9.0,
    )

    assert len(rows) == 1
    assert rows[0]["sequence_id"] == "seq_livox"
    assert abs(rows[0]["time_s"] - 5.10001) < 1.0e-12
    assert rows[0]["x_m"] == 1.0
    assert rows[0]["y_m"] == 2.0
    assert rows[0]["z_m"] == 3.0
    assert rows[0]["livox_point_index"] == 0
    assert rows[0]["intensity"] == 42
    assert rows[0]["livox_offset_time"] == 10_000
    assert rows[0]["livox_tag"] == 7
    assert rows[0]["livox_line"] == 2


def test_native_ros_extraction_supports_livox_custom_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_livox_native.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_livox_native",
                "exports": [
                    {
                        "topic": "/livox/lidar",
                        "kind": "livox_custommsg_candidate",
                        "source": "livox",
                        "voxel_size_m": 1.0,
                        "min_points": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    connection = SimpleNamespace(
        topic="/livox/lidar",
        msgtype="livox_ros_driver2/msg/CustomMsg",
    )
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=8, nanosec=0)),
        points=[
            SimpleNamespace(x=1.0, y=2.0, z=3.0, reflectivity=20),
            SimpleNamespace(x=1.2, y=2.0, z=3.0, reflectivity=25),
            SimpleNamespace(x=1.1, y=2.1, z=3.0, reflectivity=30),
        ],
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [connection]
            return [(connection, 8_000_000_000, b"livox")]

        def deserialize(self, rawdata, msgtype):
            assert rawdata == b"livox"
            assert msgtype == "livox_ros_driver2/msg/CustomMsg"
            return message

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is not None
    rows = extracted.candidates.rows
    assert len(rows) == 1
    assert rows.loc[0, "sequence_id"] == "seq_livox_native"
    assert rows.loc[0, "source"] == "livox"
    assert abs(float(rows.loc[0, "x_m"]) - 1.1) < 1.0e-9
    assert abs(float(rows.loc[0, "y_m"]) - 2.033333333333333) < 1.0e-9
    assert abs(float(rows.loc[0, "z_m"]) - 3.0) < 1.0e-9
    assert rows.loc[0, "confidence"] == 3.0
    assert extracted.manifest["candidate_rows"] == 1
    assert extracted.manifest["extracted_messages"][0]["status"] == "extracted"
    assert extracted.manifest["extracted_messages"][0]["kind"] == "livox_custommsg_candidate"
    assert (output / "native_ros_candidates.csv").exists()


def test_native_ros_radar_polar_message_to_rows_accepts_parallel_arrays() -> None:
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=3, nanosec=250_000_000)),
        ranges=[10.0, 20.0],
        azimuths_deg=[90.0, 0.0],
        elevations_deg=[0.0, 30.0],
        scores=[0.75, 0.5],
        track_ids=["east", "up"],
    )

    rows = radar_polar_message_to_rows(
        message,
        sequence_id="seq_radar_arrays",
        time_s=9.0,
        angle_unit="rad",
    )

    assert len(rows) == 2
    assert rows[0]["sequence_id"] == "seq_radar_arrays"
    assert rows[0]["time_s"] == 3.25
    assert rows[0]["range_m"] == 10.0
    assert abs(rows[0]["azimuth"] - np.pi / 2.0) < 1.0e-12
    assert rows[0]["confidence"] == 0.75
    assert rows[0]["track_id"] == "east"
    assert rows[1]["range_m"] == 20.0
    assert abs(rows[1]["elevation"] - np.pi / 6.0) < 1.0e-12


def test_native_ros_extraction_supports_radar_polar_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mcap = tmp_path / "seq_radar_native.mcap"
    mcap.write_bytes(b"fake-reader-does-not-open-this")
    topic_map = tmp_path / "topic_map.json"
    output = tmp_path / "native_out"
    topic_map.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "sequence_id": "seq_radar_native",
                "exports": [
                    {
                        "topic": "/mmwave/range_azimuth",
                        "kind": "radar_polar_candidate",
                        "source": "mmwave",
                        "angle_unit": "rad",
                        "range_std_m": 3.0,
                        "angle_std_deg": 1.0,
                        "z_std_m": 4.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    connection = SimpleNamespace(
        topic="/mmwave/range_azimuth",
        msgtype="custom_msgs/msg/RadarPolarArray",
    )
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=4, nanosec=0)),
        detections=[
            SimpleNamespace(
                range=10.0,
                azimuth=np.pi / 2.0,
                elevation=0.0,
                score=0.6,
                id="radar-a",
                label="uav",
            )
        ],
    )

    class FakeAnyReader:
        def __init__(self, paths):
            assert paths == [mcap]
            self.connections = [connection]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, *, connections):
            assert connections == [connection]
            return [(connection, 4_000_000_000, b"radar")]

        def deserialize(self, rawdata, msgtype):
            assert rawdata == b"radar"
            assert msgtype == "custom_msgs/msg/RadarPolarArray"
            return message

    monkeypatch.setitem(sys.modules, "rosbags", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "rosbags.highlevel",
        SimpleNamespace(AnyReader=FakeAnyReader),
    )

    extracted = extract_native_rosbag_topic_map(
        bag_path=mcap,
        topic_map_json=topic_map,
        output_dir=output,
    )

    assert extracted.candidates is not None
    rows = extracted.candidates.rows
    assert len(rows) == 1
    assert rows.loc[0, "sequence_id"] == "seq_radar_native"
    assert rows.loc[0, "source"] == "mmwave"
    assert rows.loc[0, "track_id"] == "radar-a"
    assert abs(float(rows.loc[0, "x_m"]) - 10.0) < 1.0e-9
    assert abs(float(rows.loc[0, "y_m"])) < 1.0e-9
    assert abs(float(rows.loc[0, "z_m"])) < 1.0e-9
    assert rows.loc[0, "std_xy_m"] == 3.0
    assert rows.loc[0, "std_z_m"] == 4.0
    assert rows.loc[0, "confidence"] == 0.6
    assert rows.loc[0, "class_name"] == "uav"
    assert extracted.manifest["candidate_rows"] == 1
    assert extracted.manifest["extracted_messages"][0]["status"] == "extracted"
    assert extracted.manifest["extracted_messages"][0]["kind"] == "radar_polar_candidate"
    assert (output / "native_ros_candidates.csv").exists()


def test_ros2_topic_map_template_infers_polar_radar_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /mmwave/range_azimuth",
                "        type: custom_msgs/msg/RadarPolarArray",
                "      message_count: 3",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    export = payload["exports"][0]
    assert export["kind"] == "radar_polar_candidate"
    assert export["source"] == "mmwave_range_azimuth"
    assert export["column_aliases"]["range"] == "range_m"
    assert export["column_aliases"]["bearing"] == "azimuth_deg"
    assert export["column_aliases"]["el"] == "elevation_deg"


def test_ros2_topic_map_template_infers_point_and_transform_topics(
    tmp_path: Path,
) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/point",
                "        type: geometry_msgs/msg/PointStamped",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/transform",
                "        type: geometry_msgs/msg/TransformStamped",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "point_candidate",
        "transform_truth",
    ]
    assert payload["exports"][0]["source"] == "detector_point"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_tf_message_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /tf",
                "        type: tf2_msgs/msg/TFMessage",
                "      message_count: 4",
                "    - topic_metadata:",
                "        name: /ground_truth/tf",
                "        type: tf2_msgs/msg/TFMessage",
                "      message_count: 4",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "tf_candidate",
        "tf_truth",
    ]
    assert payload["exports"][0]["source"] == "tf"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_path_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/path",
                "        type: nav_msgs/msg/Path",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/path",
                "        type: nav_msgs/msg/Path",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "path_candidate",
        "path_truth",
    ]
    assert payload["exports"][0]["source"] == "detector_path"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_pose_array_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/poses",
                "        type: geometry_msgs/msg/PoseArray",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/poses",
                "        type: geometry_msgs/msg/PoseArray",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "pose_array_candidate",
        "pose_array_truth",
    ]
    assert payload["exports"][0]["source"] == "detector_poses"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_pose_covariance_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/pose_cov",
                "        type: geometry_msgs/msg/PoseWithCovarianceStamped",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/pose",
                "        type: geometry_msgs/msg/Pose",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "pose_candidate",
        "pose_truth",
    ]
    assert payload["exports"][0]["source"] == "detector_pose_cov"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_geodetic_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /gps/fix",
                "        type: sensor_msgs/msg/NavSatFix",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/geopose",
                "        type: geographic_msgs/msg/GeoPoseStamped",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "navsatfix_candidate",
        "geopose_truth",
    ]
    assert payload["exports"][0]["source"] == "gps_fix"
    assert payload["exports"][1]["source"] is None
    assert payload["exports"][0]["enu_origin_lla"] == "LAT,LON,ALT"
    assert payload["exports"][1]["enu_origin_lla"] == "LAT,LON,ALT"


def test_ros2_topic_map_template_infers_plain_point_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/point",
                "        type: geometry_msgs/msg/Point",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/point",
                "        type: geometry_msgs/msg/Point",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "point_candidate",
        "point_truth",
    ]
    assert payload["exports"][0]["source"] == "detector_point"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_detection3d_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/detections",
                "        type: vision_msgs/msg/Detection3DArray",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/detection",
                "        type: vision_msgs/msg/Detection3D",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "detection3d_array_candidate",
        "detection3d_truth",
    ]
    assert payload["exports"][0]["source"] == "detector_detections"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_bounding_box3d_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/boxes",
                "        type: vision_msgs/msg/BoundingBox3DArray",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/box",
                "        type: jsk_recognition_msgs/msg/BoundingBox",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "bounding_box3d_array_candidate",
        "bounding_box3d_truth",
    ]
    assert payload["exports"][0]["source"] == "detector_boxes"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_tracked_object_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /perception/tracked_objects",
                "        type: autoware_auto_perception_msgs/msg/TrackedObjects",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/detected_objects",
                "        type: autoware_auto_perception_msgs/msg/DetectedObjectArray",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "tracked_objects_candidate",
        "tracked_objects_truth",
    ]
    assert payload["exports"][0]["source"] == "perception_tracked_objects"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_image_timestamp_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /camera/front/image_raw",
                "        type: sensor_msgs/msg/Image",
                "      message_count: 4",
                "    - topic_metadata:",
                "        name: /camera/front/image_compressed",
                "        type: sensor_msgs/msg/CompressedImage",
                "      message_count: 4",
                "    - topic_metadata:",
                "        name: /camera/front/camera_info",
                "        type: sensor_msgs/msg/CameraInfo",
                "      message_count: 1",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(
        report,
        tmp_path / "topic_map_template.json",
        template_mode="native",
    )
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "image_timestamps",
        "image_timestamps",
        "camera_info_calibration",
    ]
    assert payload["exports"][0]["source"] == "camera_front_image_raw"
    assert payload["exports"][1]["source"] == "camera_front_image_compressed"
    assert all("path" not in entry for entry in payload["exports"])


def test_ros2_topic_map_template_infers_audio_timestamp_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /microphone/audio",
                "        type: audio_common_msgs/msg/AudioData",
                "      message_count: 3",
                "    - topic_metadata:",
                "        name: /array/audio_stamped",
                "        type: audio_common_msgs/msg/AudioStamped",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(
        report,
        tmp_path / "topic_map_template.json",
        template_mode="native",
    )
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "audio_timestamps",
        "audio_timestamps",
    ]
    assert payload["exports"][0]["source"] == "microphone_audio"
    assert payload["exports"][1]["source"] == "array_audio_stamped"
    assert all("path" not in entry for entry in payload["exports"])


def test_ros2_topic_map_template_infers_imu_timestamp_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /os1_cloud_node1/imu",
                "        type: sensor_msgs/msg/Imu",
                "      message_count: 5",
                "    - topic_metadata:",
                "        name: /os1_cloud_node2/imu",
                "        type: sensor_msgs/msg/Imu",
                "      message_count: 5",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(
        report,
        tmp_path / "topic_map_template.json",
        template_mode="native",
    )
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "imu_timestamps",
        "imu_timestamps",
    ]
    assert payload["exports"][0]["source"] == "os1_cloud_node1_imu"
    assert payload["exports"][1]["source"] == "os1_cloud_node2_imu"
    assert all("path" not in entry for entry in payload["exports"])


def test_ros2_topic_map_template_infers_detection2d_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /camera/detections2d",
                "        type: vision_msgs/msg/Detection2DArray",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /camera/best_detection",
                "        type: vision_msgs/msg/Detection2D",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /camera/camera_info",
                "        type: sensor_msgs/msg/CameraInfo",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "camera_detections_candidate",
        "camera_detections_candidate",
        "camera_info_calibration",
    ]
    export = payload["exports"][0]
    assert export["kind"] == "camera_detections_candidate"
    assert export["source"] == "camera_detections2d"
    assert export["camera_calibration_file"] == "PATH/TO/camera_info.json"
    assert export["column_aliases"]["center_x"] == "u_px"
    assert export["column_aliases"]["center_y"] == "v_px"
    singular_export = payload["exports"][1]
    assert singular_export["kind"] == "camera_detections_candidate"
    assert singular_export["source"] == "camera_best_detection"
    assert singular_export["camera_calibration_file"] == "PATH/TO/camera_info.json"
    assert payload["exports"][2]["source"] == "camera_camera_info"


def test_ros2_topic_map_template_infers_marker_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/markers",
                "        type: visualization_msgs/msg/MarkerArray",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/marker",
                "        type: visualization_msgs/msg/Marker",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "marker_array_candidate",
        "marker_truth",
    ]
    assert payload["exports"][0]["source"] == "detector_markers"
    assert payload["exports"][1]["source"] is None


def test_ros2_topic_map_template_infers_multidof_topics(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /detector/multidof_state",
                "        type: sensor_msgs/msg/MultiDOFJointState",
                "      message_count: 2",
                "    - topic_metadata:",
                "        name: /ground_truth/multidof_trajectory",
                "        type: trajectory_msgs/msg/MultiDOFJointTrajectory",
                "      message_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_rosbag(bag)
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))

    assert [entry["kind"] for entry in payload["exports"]] == [
        "multidof_joint_state_candidate",
        "multidof_joint_trajectory_truth",
    ]
    assert payload["exports"][0]["source"] == "detector_multidof_state"
    assert payload["exports"][1]["source"] is None


def test_native_ros_position_message_to_row_accepts_common_position_messages() -> None:
    point_message = SimpleNamespace(point=SimpleNamespace(x=1.0, y=2.0, z=3.0))
    transform_message = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=4.0, y=5.0, z=6.0)
        )
    )
    pose_message = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=SimpleNamespace(x=7.0, y=8.0, z=9.0))
        )
    )
    exact_pose_message = SimpleNamespace(
        position=SimpleNamespace(x=10.0, y=11.0, z=12.0)
    )
    covariance_pose_message = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=SimpleNamespace(x=13.0, y=14.0, z=15.0)),
            covariance=[0.0] * 36,
        )
    )

    point_row = position_message_to_row(point_message, sequence_id="seq", time_s=1.25)
    transform_row = position_message_to_row(
        transform_message,
        sequence_id="seq",
        time_s=2.5,
    )
    pose_row = position_message_to_row(pose_message, sequence_id="seq", time_s=3.75)
    exact_pose_row = position_message_to_row(
        exact_pose_message,
        sequence_id="seq",
        time_s=4.25,
    )
    covariance_pose_row = position_message_to_row(
        covariance_pose_message,
        sequence_id="seq",
        time_s=5.5,
    )

    assert point_row == {
        "sequence_id": "seq",
        "time_s": 1.25,
        "x_m": 1.0,
        "y_m": 2.0,
        "z_m": 3.0,
    }
    assert transform_row["x_m"] == 4.0
    assert transform_row["time_s"] == 2.5
    assert pose_row["z_m"] == 9.0
    assert exact_pose_row["x_m"] == 10.0
    assert exact_pose_row["time_s"] == 4.25
    assert covariance_pose_row["z_m"] == 15.0


def test_native_ros_position_message_to_rows_filters_tf_message_transforms() -> None:
    stamp = SimpleNamespace(sec=10, nanosec=250_000_000)
    tf_message = SimpleNamespace(
        transforms=[
            SimpleNamespace(
                header=SimpleNamespace(stamp=stamp, frame_id="world"),
                child_frame_id="uav",
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=1.0, y=2.0, z=3.0)
                ),
            ),
            SimpleNamespace(
                header=SimpleNamespace(frame_id="world"),
                child_frame_id="camera",
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=4.0, y=5.0, z=6.0)
                ),
            ),
        ]
    )

    rows = position_message_to_rows(
        tf_message,
        sequence_id="seq_tf",
        time_s=1.0,
        child_frame_id="uav",
    )

    assert len(rows) == 1
    assert rows[0]["sequence_id"] == "seq_tf"
    assert rows[0]["time_s"] == 10.25
    assert rows[0]["x_m"] == 1.0
    assert rows[0]["child_frame_id"] == "uav"
    assert rows[0]["frame_id"] == "world"


def test_native_ros_position_message_to_rows_expands_path_poses() -> None:
    path_message = SimpleNamespace(
        poses=[
            SimpleNamespace(
                header=SimpleNamespace(
                    stamp=SimpleNamespace(sec=20, nanosec=0),
                    frame_id="world",
                ),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0, z=3.0)
                ),
            ),
            SimpleNamespace(
                header=SimpleNamespace(
                    stamp=SimpleNamespace(sec=21, nanosec=500_000_000),
                    frame_id="camera",
                ),
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=4.0, y=5.0, z=6.0)
                ),
            ),
        ]
    )

    rows = position_message_to_rows(
        path_message,
        sequence_id="seq_path",
        time_s=1.0,
        frame_id="world",
    )

    assert len(rows) == 1
    assert rows[0]["sequence_id"] == "seq_path"
    assert rows[0]["time_s"] == 20.0
    assert rows[0]["x_m"] == 1.0
    assert rows[0]["frame_id"] == "world"


def test_native_ros_position_message_to_rows_expands_pose_array_parent_header() -> None:
    pose_array = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=30, nanosec=125_000_000),
            frame_id="world",
        ),
        poses=[
            SimpleNamespace(position=SimpleNamespace(x=1.0, y=2.0, z=3.0)),
            SimpleNamespace(position=SimpleNamespace(x=4.0, y=5.0, z=6.0)),
        ],
    )

    rows = position_message_to_rows(
        pose_array,
        sequence_id="seq_pose_array",
        time_s=1.0,
        frame_id="world",
    )

    assert len(rows) == 2
    assert rows[0]["sequence_id"] == "seq_pose_array"
    assert rows[0]["time_s"] == 30.125
    assert rows[0]["x_m"] == 1.0
    assert rows[0]["frame_id"] == "world"
    assert rows[1]["z_m"] == 6.0


def test_native_ros_geodetic_message_to_rows_projects_navsatfix() -> None:
    projector = LocalENUProjector(35.0, -78.0, 100.0)
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=35, nanosec=500_000_000),
            frame_id="gps",
        ),
        latitude=35.0,
        longitude=-78.0,
        altitude=112.0,
        position_covariance=[
            4.0,
            0.0,
            0.0,
            0.0,
            9.0,
            0.0,
            0.0,
            0.0,
            16.0,
        ],
        position_covariance_type=2,
    )

    rows = geodetic_message_to_rows(
        message,
        sequence_id="seq_gps",
        time_s=1.0,
        projector=projector,
        frame_id="gps",
    )

    assert len(rows) == 1
    assert rows[0]["sequence_id"] == "seq_gps"
    assert rows[0]["time_s"] == 35.5
    np.testing.assert_allclose(
        [rows[0]["x_m"], rows[0]["y_m"], rows[0]["z_m"]],
        [0.0, 0.0, 12.0],
        atol=1.0e-6,
    )
    assert rows[0]["latitude_deg"] == 35.0
    assert rows[0]["longitude_deg"] == -78.0
    assert rows[0]["altitude_m"] == 112.0
    assert rows[0]["std_xy_m"] == 3.0
    assert rows[0]["std_z_m"] == 4.0
    assert rows[0]["navsat_covariance_type"] == "2"


def test_native_ros_geodetic_message_to_rows_accepts_geopose() -> None:
    projector = LocalENUProjector(35.0, -78.0, 100.0)
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id="world"),
        pose=SimpleNamespace(
            position=SimpleNamespace(latitude=35.0, longitude=-78.0, altitude=105.0)
        ),
    )

    rows = geodetic_message_to_rows(
        message,
        sequence_id="seq_geo",
        time_s=2.0,
        projector=projector,
    )

    assert len(rows) == 1
    assert rows[0]["time_s"] == 2.0
    assert rows[0]["frame_id"] == "world"
    np.testing.assert_allclose(
        [rows[0]["x_m"], rows[0]["y_m"], rows[0]["z_m"]],
        [0.0, 0.0, 5.0],
        atol=1.0e-6,
    )


def test_native_ros_detection2d_message_to_rows_extracts_bbox_centers() -> None:
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=20, nanosec=500_000_000),
            frame_id="cam0",
        ),
        detections=[
            SimpleNamespace(
                header=SimpleNamespace(
                    stamp=SimpleNamespace(sec=21, nanosec=250_000_000),
                    frame_id="cam0",
                ),
                id="det-7",
                bbox=SimpleNamespace(
                    center=SimpleNamespace(x=60.0, y=50.0),
                    size_x=8.0,
                    size_y=4.0,
                ),
                results=[
                    SimpleNamespace(
                        hypothesis=SimpleNamespace(class_id="drone", score=0.85)
                    )
                ],
            ),
            SimpleNamespace(
                header=SimpleNamespace(frame_id="other_camera"),
                bbox=SimpleNamespace(center=SimpleNamespace(x=10.0, y=10.0)),
            ),
        ],
    )

    rows = detection2d_message_to_rows(
        message,
        sequence_id="seq_det2d",
        time_s=1.0,
        frame_id="cam0",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["sequence_id"] == "seq_det2d"
    assert row["time_s"] == 21.25
    assert row["u_px"] == 60.0
    assert row["v_px"] == 50.0
    assert row["x1"] == 56.0
    assert row["y1"] == 48.0
    assert row["x2"] == 64.0
    assert row["y2"] == 52.0
    assert row["track_id"] == "det-7"
    assert row["confidence"] == 0.85
    assert row["class_name"] == "drone"
    assert row["frame_id"] == "cam0"


def test_native_ros_detection2d_message_to_rows_uses_highest_score_result() -> None:
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id="cam0"),
        bbox=SimpleNamespace(center=SimpleNamespace(x=60.0, y=50.0)),
        results=[
            SimpleNamespace(hypothesis=SimpleNamespace(class_id="bird", score=0.2)),
            SimpleNamespace(hypothesis=SimpleNamespace(class_id="uav", score=0.9)),
        ],
    )

    rows = detection2d_message_to_rows(
        message,
        sequence_id="seq_det2d_best",
        time_s=1.0,
        frame_id="cam0",
    )

    assert len(rows) == 1
    assert rows[0]["confidence"] == 0.9
    assert rows[0]["class_name"] == "uav"


def test_native_ros_detection3d_message_to_rows_extracts_bbox_centers() -> None:
    detections = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=40, nanosec=500_000_000),
            frame_id="world",
        ),
        detections=[
            SimpleNamespace(
                id="det-1",
                bbox=SimpleNamespace(
                    center=SimpleNamespace(
                        position=SimpleNamespace(x=1.0, y=2.0, z=3.0)
                    )
                ),
                results=[
                    SimpleNamespace(
                        hypothesis=SimpleNamespace(class_id="uav", score=0.8)
                    )
                ],
            ),
            SimpleNamespace(
                header=SimpleNamespace(frame_id="camera"),
                id="det-2",
                bbox=SimpleNamespace(
                    center=SimpleNamespace(
                        position=SimpleNamespace(x=4.0, y=5.0, z=6.0)
                    )
                ),
                results=[SimpleNamespace(class_id="bird", score=0.2)],
            ),
        ],
    )

    rows = detection3d_message_to_rows(
        detections,
        sequence_id="seq_detection",
        time_s=1.0,
        frame_id="world",
    )

    assert len(rows) == 1
    assert rows[0]["sequence_id"] == "seq_detection"
    assert rows[0]["time_s"] == 40.5
    assert rows[0]["x_m"] == 1.0
    assert rows[0]["y_m"] == 2.0
    assert rows[0]["z_m"] == 3.0
    assert rows[0]["frame_id"] == "world"
    assert rows[0]["detection_id"] == "det-1"
    assert rows[0]["confidence"] == 0.8
    assert rows[0]["class_name"] == "uav"


def test_native_ros_detection3d_message_to_rows_uses_highest_score_result() -> None:
    message = SimpleNamespace(
        header=SimpleNamespace(frame_id="world"),
        bbox=SimpleNamespace(
            center=SimpleNamespace(position=SimpleNamespace(x=1.0, y=2.0, z=3.0))
        ),
        results=[
            SimpleNamespace(hypothesis=SimpleNamespace(class_id="bird", score=0.25)),
            SimpleNamespace(hypothesis=SimpleNamespace(class_id="uav", score=0.85)),
        ],
    )

    rows = detection3d_message_to_rows(
        message,
        sequence_id="seq_det3d_best",
        time_s=1.0,
        frame_id="world",
    )

    assert len(rows) == 1
    assert rows[0]["confidence"] == 0.85
    assert rows[0]["class_name"] == "uav"


def test_native_ros_bounding_box3d_message_to_rows_extracts_centers() -> None:
    boxes = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=41, nanosec=250_000_000),
            frame_id="world",
        ),
        boxes=[
            SimpleNamespace(
                id="box-1",
                center=SimpleNamespace(
                    position=SimpleNamespace(x=7.0, y=8.0, z=9.0)
                ),
                size=SimpleNamespace(x=1.5, y=2.5, z=3.5),
                label="quadrotor",
                confidence=0.77,
            ),
            SimpleNamespace(
                header=SimpleNamespace(frame_id="camera"),
                id="box-2",
                center=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0, z=3.0)
                ),
            ),
        ],
    )

    rows = bounding_box3d_message_to_rows(
        boxes,
        sequence_id="seq_boxes",
        time_s=1.0,
        frame_id="world",
    )

    assert len(rows) == 1
    assert rows[0]["sequence_id"] == "seq_boxes"
    assert rows[0]["time_s"] == 41.25
    assert rows[0]["x_m"] == 7.0
    assert rows[0]["y_m"] == 8.0
    assert rows[0]["z_m"] == 9.0
    assert rows[0]["frame_id"] == "world"
    assert rows[0]["box_index"] == 0
    assert rows[0]["box_id"] == "box-1"
    assert rows[0]["track_id"] == "box-1"
    assert rows[0]["class_name"] == "quadrotor"
    assert rows[0]["confidence"] == 0.77
    assert rows[0]["box_size_x_m"] == 1.5
    assert rows[0]["box_size_y_m"] == 2.5
    assert rows[0]["box_size_z_m"] == 3.5


def test_native_ros_bounding_box3d_message_to_rows_accepts_single_box() -> None:
    box = {
        "header": {"frame_id": "world"},
        "box_id": "single-box",
        "center": {"position": {"x": 1.0, "y": 2.0, "z": 3.0}},
        "dimensions": {"x": 4.0, "y": 5.0, "z": 6.0},
    }

    rows = bounding_box3d_message_to_rows(
        box,
        sequence_id="seq_single_box",
        time_s=12.0,
        frame_id="world",
    )

    assert len(rows) == 1
    assert rows[0]["time_s"] == 12.0
    assert rows[0]["box_id"] == "single-box"
    assert rows[0]["x_m"] == 1.0
    assert rows[0]["box_size_z_m"] == 6.0


def test_native_ros_tracked_objects_message_to_rows_extracts_kinematics() -> None:
    covariance = [0.0] * 36
    covariance[0] = 0.25
    covariance[7] = 0.36
    covariance[14] = 0.49
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=80, nanosec=250_000_000),
            frame_id="world",
        ),
        objects=[
            SimpleNamespace(
                object_id=SimpleNamespace(uuid=[1, 2, 3, 4]),
                kinematics=SimpleNamespace(
                    pose_with_covariance=SimpleNamespace(
                        pose=SimpleNamespace(
                            position=SimpleNamespace(x=10.0, y=20.0, z=30.0)
                        ),
                        covariance=covariance,
                    )
                ),
                classification=SimpleNamespace(label="quadrotor", probability=0.91),
            ),
            SimpleNamespace(
                header=SimpleNamespace(frame_id="camera"),
                track_id="filtered",
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0, z=3.0)
                ),
            ),
        ],
    )

    rows = tracked_objects_message_to_rows(
        message,
        sequence_id="seq_objects",
        time_s=1.0,
        frame_id="world",
    )

    assert len(rows) == 1
    assert rows[0]["sequence_id"] == "seq_objects"
    assert rows[0]["time_s"] == 80.25
    assert rows[0]["x_m"] == 10.0
    assert rows[0]["y_m"] == 20.0
    assert rows[0]["z_m"] == 30.0
    assert rows[0]["frame_id"] == "world"
    assert rows[0]["object_index"] == 0
    assert rows[0]["object_id"] == "01020304"
    assert rows[0]["track_id"] == "01020304"
    assert rows[0]["class_name"] == "quadrotor"
    assert rows[0]["confidence"] == 0.91
    assert rows[0]["std_xy_m"] == 0.6
    assert rows[0]["std_z_m"] == 0.7


def test_native_ros_marker_message_to_rows_extracts_marker_positions() -> None:
    marker_array = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=50, nanosec=250_000_000),
            frame_id="world",
        ),
        markers=[
            SimpleNamespace(
                ns="uav",
                id=7,
                type=2,
                action=0,
                text="quadrotor",
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0, z=3.0)
                ),
            ),
            SimpleNamespace(
                header=SimpleNamespace(frame_id="camera"),
                ns="uav",
                id=8,
                action=0,
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=4.0, y=5.0, z=6.0)
                ),
            ),
            SimpleNamespace(
                ns="old",
                id=9,
                action=2,
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=7.0, y=8.0, z=9.0)
                ),
            ),
        ],
    )

    rows = marker_message_to_rows(
        marker_array,
        sequence_id="seq_marker",
        time_s=1.0,
        frame_id="world",
    )

    assert len(rows) == 1
    assert rows[0]["sequence_id"] == "seq_marker"
    assert rows[0]["time_s"] == 50.25
    assert rows[0]["x_m"] == 1.0
    assert rows[0]["frame_id"] == "world"
    assert rows[0]["marker_namespace"] == "uav"
    assert rows[0]["marker_id"] == "7"
    assert rows[0]["marker_track_id"] == "uav:7"
    assert rows[0]["marker_type"] == "2"
    assert rows[0]["class_name"] == "quadrotor"


def test_native_ros_marker_message_to_rows_uses_point_centroid() -> None:
    marker = SimpleNamespace(
        header=SimpleNamespace(frame_id="world"),
        ns="points",
        id=3,
        points=[
            SimpleNamespace(x=1.0, y=2.0, z=3.0),
            SimpleNamespace(x=3.0, y=4.0, z=5.0),
        ],
    )

    rows = marker_message_to_rows(marker, sequence_id="seq_marker", time_s=6.0)

    assert len(rows) == 1
    assert rows[0]["time_s"] == 6.0
    assert rows[0]["x_m"] == 2.0
    assert rows[0]["y_m"] == 3.0
    assert rows[0]["z_m"] == 4.0
    assert rows[0]["marker_track_id"] == "points:3"


def test_native_ros_multidof_message_to_rows_expands_joint_state() -> None:
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=60, nanosec=125_000_000),
            frame_id="world",
        ),
        joint_names=["uav", "payload"],
        transforms=[
            SimpleNamespace(translation=SimpleNamespace(x=1.0, y=2.0, z=3.0)),
            SimpleNamespace(translation=SimpleNamespace(x=4.0, y=5.0, z=6.0)),
        ],
    )

    rows = multidof_message_to_rows(
        message,
        sequence_id="seq_multidof",
        time_s=1.0,
        frame_id="world",
    )

    assert len(rows) == 2
    assert rows[0]["sequence_id"] == "seq_multidof"
    assert rows[0]["time_s"] == 60.125
    assert rows[0]["x_m"] == 1.0
    assert rows[0]["frame_id"] == "world"
    assert rows[0]["joint_name"] == "uav"
    assert rows[0]["multidof_transform_index"] == 0
    assert rows[1]["joint_name"] == "payload"
    assert rows[1]["z_m"] == 6.0


def test_native_ros_multidof_message_to_rows_expands_trajectory_points() -> None:
    trajectory = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=70, nanosec=0),
            frame_id="world",
        ),
        joint_names=["uav"],
        points=[
            SimpleNamespace(
                time_from_start=SimpleNamespace(sec=1, nanosec=500_000_000),
                transforms=[
                    SimpleNamespace(
                        translation=SimpleNamespace(x=1.0, y=2.0, z=3.0)
                    )
                ],
            ),
            SimpleNamespace(
                time_from_start=SimpleNamespace(sec=3, nanosec=0),
                transforms=[
                    SimpleNamespace(
                        translation=SimpleNamespace(x=4.0, y=5.0, z=6.0)
                    )
                ],
            ),
        ],
    )

    rows = multidof_message_to_rows(
        trajectory,
        sequence_id="seq_multidof_traj",
        time_s=1.0,
        frame_id="world",
    )

    assert len(rows) == 2
    assert rows[0]["time_s"] == 71.5
    assert rows[0]["joint_name"] == "uav"
    assert rows[0]["multidof_point_index"] == 0
    assert rows[1]["time_s"] == 73.0
    assert rows[1]["x_m"] == 4.0
    assert rows[1]["multidof_point_index"] == 1


def test_pointcloud2_decoder_and_candidate_clustering() -> None:
    fields = [
        SimpleNamespace(name="x", offset=0, datatype=7, count=1),
        SimpleNamespace(name="y", offset=4, datatype=7, count=1),
        SimpleNamespace(name="z", offset=8, datatype=7, count=1),
    ]
    points = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.1], [0.2, 0.1, 1.0]],
        dtype="<f4",
    )
    msg = SimpleNamespace(
        fields=fields,
        data=points.tobytes(),
        width=3,
        height=1,
        point_step=12,
        is_bigendian=False,
        is_dense=True,
    )

    decoded = pointcloud2_to_dataframe(msg)
    candidates = pointcloud2_to_candidates(
        msg,
        sequence_id="seq_pc2",
        time_s=7.5,
        source="livox",
        voxel_size_m=0.5,
        min_points=3,
    )

    assert decoded.shape == (3, 3)
    assert len(candidates.rows) == 1
    assert candidates.rows.loc[0, "sequence_id"] == "seq_pc2"
    assert candidates.rows.loc[0, "source"] == "livox"


def test_pointcloud2_decoder_respects_organized_cloud_row_step_padding() -> None:
    fields = [
        SimpleNamespace(name="x", offset=0, datatype=7, count=1),
        SimpleNamespace(name="y", offset=4, datatype=7, count=1),
        SimpleNamespace(name="z", offset=8, datatype=7, count=1),
        SimpleNamespace(name="intensity", offset=12, datatype=7, count=1),
    ]
    point_step = 16
    row_step = 40
    data = bytearray(row_step * 2)
    points = [
        (0, 0, (0.0, 0.0, 1.0, 10.0)),
        (0, 1, (1.0, 0.0, 1.0, 20.0)),
        (1, 0, (0.0, 1.0, 2.0, 30.0)),
        (1, 1, (1.0, 1.0, 2.0, 40.0)),
    ]
    for row, column, values in points:
        struct.pack_into("<ffff", data, row * row_step + column * point_step, *values)
    msg = SimpleNamespace(
        fields=fields,
        data=bytes(data),
        width=2,
        height=2,
        point_step=point_step,
        row_step=row_step,
        is_bigendian=False,
        is_dense=True,
    )

    decoded = pointcloud2_to_dataframe(msg)

    assert decoded[["x_m", "y_m", "z_m"]].values.tolist() == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 2.0],
    ]
    assert decoded["intensity"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_calibration_file_accepts_yaml_json_subset(tmp_path: Path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text(
        json.dumps(
            {
                "world_frame": "test",
                "sensors": {
                    "radar": {
                        "translation_m": [1.0, 2.0, 3.0],
                        "rpy_deg": [0.0, 0.0, 0.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = load_calibration_file(path)

    assert calibration.world_frame == "test"
    assert calibration.get("radar") is not None


def test_calibration_file_accepts_transform_matrix_alias(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "world_frame": "test",
                "sensors": {
                    "radar": {
                        "T_sensor_to_world": {
                            "rows": 4,
                            "cols": 4,
                            "data": [
                                1.0,
                                0.0,
                                0.0,
                                10.0,
                                0.0,
                                1.0,
                                0.0,
                                20.0,
                                0.0,
                                0.0,
                                1.0,
                                30.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                            ],
                        },
                        "time_offset_s": 0.25,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = load_calibration_file(path)
    transformed = transform_candidate_frame(
        CandidateFrame(
            pd.DataFrame(
                {
                    "sequence_id": ["seq"],
                    "time_s": [1.0],
                    "source": ["radar"],
                    "x_m": [1.0],
                    "y_m": [2.0],
                    "z_m": [3.0],
                }
            )
        ),
        calibration,
    )
    row = transformed.rows.iloc[0]
    assert (row["x_m"], row["y_m"], row["z_m"]) == (11.0, 22.0, 33.0)
    assert abs(float(row["time_s"]) - 1.25) < 1.0e-9


def test_radar_polar_csv_converts_to_candidates(tmp_path: Path) -> None:
    from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates

    radar = tmp_path / "radar_polar.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "range_m": [10.0, 10.0],
            "azimuth_deg": [0.0, 90.0],
            "elevation_deg": [0.0, 0.0],
            "track_id": ["r1", "r1"],
            "confidence": [0.8, 0.9],
        }
    ).to_csv(radar, index=False)

    candidates = load_radar_polar_csv_as_candidates(
        radar,
        azimuth_convention="north-clockwise",
    )

    assert len(candidates.rows) == 2
    first = candidates.rows.iloc[0]
    second = candidates.rows.iloc[1]
    assert abs(first["x_m"]) < 1.0e-9
    assert abs(first["y_m"] - 10.0) < 1.0e-9
    assert abs(second["x_m"] - 10.0) < 1.0e-9


def test_radar_polar_tsv_converts_to_candidates(tmp_path: Path) -> None:
    from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates

    radar = tmp_path / "radar_polar.tsv"
    pd.DataFrame(
        {
            "timestamp": [0.0],
            "range": [10.0],
            "bearing_deg": [90.0],
            "track": ["r1"],
        }
    ).to_csv(radar, sep="\t", index=False)

    candidates = load_radar_polar_csv_as_candidates(
        radar,
        sequence_id="seq_tsv_radar",
        azimuth_convention="north-clockwise",
    )

    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_tsv_radar"
    assert row["track_id"] == "r1"
    assert abs(float(row["x_m"]) - 10.0) < 1.0e-9


def test_radar_polar_json_converts_to_candidates(tmp_path: Path) -> None:
    from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates

    radar = tmp_path / "radar_polar.json"
    radar.write_text(
        json.dumps(
            {
                "radar_detections": [
                    {
                        "timestamp_ms": 1250,
                        "range_m": 10.0,
                        "azimuth_deg": 90.0,
                        "track": "r1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_radar_polar_csv_as_candidates(
        radar,
        sequence_id="seq_json_radar",
        azimuth_convention="north-clockwise",
    )

    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_json_radar"
    assert row["track_id"] == "r1"
    assert abs(float(row["time_s"]) - 1.25) < 1.0e-12
    assert abs(float(row["x_m"]) - 10.0) < 1.0e-9


def test_radar_polar_json_accepts_target_wrappers(tmp_path: Path) -> None:
    from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates

    radar = tmp_path / "radar_targets.json"
    radar.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "timestamp_s": 2.0,
                        "range": 12.0,
                        "bearing_deg": 90.0,
                        "id": "target-7",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_radar_polar_csv_as_candidates(
        radar,
        sequence_id="seq_json_targets",
        azimuth_convention="north-clockwise",
    )

    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_json_targets"
    assert row["track_id"] == "target-7"
    assert abs(float(row["time_s"]) - 2.0) < 1.0e-12
    assert abs(float(row["x_m"]) - 12.0) < 1.0e-9


def test_radar_polar_json_propagates_parent_sequence_and_time(tmp_path: Path) -> None:
    from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates

    radar = tmp_path / "radar_parent_metadata.json"
    radar.write_text(
        json.dumps(
            {
                "sequence_id": "seq_parent_radar",
                "timestamp_s": 3.5,
                "radar_polar": [
                    {"range": 12.0, "bearing_deg": 90.0, "id": "target-7"},
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_radar_polar_csv_as_candidates(
        radar,
        azimuth_convention="north-clockwise",
    )

    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_parent_radar"
    assert row["track_id"] == "target-7"
    assert abs(float(row["time_s"]) - 3.5) < 1.0e-12
    assert abs(float(row["x_m"]) - 12.0) < 1.0e-9


def test_radar_polar_json_keeps_row_metadata_over_parent_defaults(tmp_path: Path) -> None:
    from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates

    radar = tmp_path / "radar_row_metadata.json"
    radar.write_text(
        json.dumps(
            {
                "sequence_id": "seq_parent_radar",
                "timestamp_s": 3.5,
                "detections": [
                    {
                        "sequence": "seq_child_radar",
                        "timestamp_s": 4.5,
                        "range": 12.0,
                        "bearing_deg": 90.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_radar_polar_csv_as_candidates(
        radar,
        azimuth_convention="north-clockwise",
    )

    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_child_radar"
    assert abs(float(row["time_s"]) - 4.5) < 1.0e-12
    assert abs(float(row["x_m"]) - 12.0) < 1.0e-9


def test_radar_polar_jsonl_converts_to_candidates(tmp_path: Path) -> None:
    from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates

    radar = tmp_path / "radar_polar.jsonl"
    rows = [
        {"timestamp_s": 0.0, "range_m": 10.0, "azimuth_deg": 0.0, "track": "r1"},
        {"timestamp_s": 1.0, "range_m": 10.0, "azimuth_deg": 90.0, "track": "r1"},
    ]
    radar.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    candidates = load_radar_polar_csv_as_candidates(
        radar,
        sequence_id="seq_jsonl_radar",
        azimuth_convention="north-clockwise",
    )

    assert candidates.rows["sequence_id"].tolist() == [
        "seq_jsonl_radar",
        "seq_jsonl_radar",
    ]
    assert candidates.rows["track_id"].tolist() == ["r1", "r1"]
    assert abs(float(candidates.rows.iloc[0]["y_m"]) - 10.0) < 1.0e-9
    assert abs(float(candidates.rows.iloc[1]["x_m"]) - 10.0) < 1.0e-9


def test_radar_polar_gzipped_jsonl_converts_to_candidates(tmp_path: Path) -> None:
    from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates

    radar = tmp_path / "radar_polar.jsonl.gz"
    rows = [
        {"timestamp_s": 0.0, "range_m": 10.0, "azimuth_deg": 90.0, "track": "r1"},
    ]
    with gzip.open(radar, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row) for row in rows))

    candidates = load_radar_polar_csv_as_candidates(
        radar,
        sequence_id="seq_jsonl_gz_radar",
        azimuth_convention="north-clockwise",
    )

    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq_jsonl_gz_radar"
    assert row["track_id"] == "r1"
    assert abs(float(row["x_m"]) - 10.0) < 1.0e-9


def test_radar_polar_loader_accepts_millisecond_timestamps(tmp_path: Path) -> None:
    from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates

    radar = tmp_path / "radar_polar.csv"
    pd.DataFrame(
        {
            "timestamp_ms": [1250],
            "range_m": [10.0],
            "azimuth_deg": [90.0],
        }
    ).to_csv(radar, index=False)

    candidates = load_radar_polar_csv_as_candidates(
        radar,
        sequence_id="seq_ms_radar",
        azimuth_convention="north-clockwise",
    )

    assert abs(float(candidates.rows.loc[0, "time_s"]) - 1.25) < 1e-12


def test_camera_detections_backproject_to_world_candidates(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                        "translation_m": [1.0, 0.0, 0.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [0.0],
            "source": ["cam0"],
            "x1": [45.0],
            "y1": [45.0],
            "x2": [55.0],
            "y2": [55.0],
            "depth_m": [10.0],
            "confidence": [0.7],
            "class_name": ["Mavic3"],
        }
    ).to_csv(detections, index=False)

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    assert len(candidates.rows) == 1
    row = candidates.rows.iloc[0]
    assert abs(row["x_m"] - 1.0) < 1.0e-9
    assert abs(row["y_m"]) < 1.0e-9
    assert abs(row["z_m"] - 10.0) < 1.0e-9
    assert row["class_name"] == "Mavic3"


def test_camera_detections_accept_sec_nanosec_timestamps(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "sec": [3],
            "nanosec": [500_000_000],
            "source": ["cam0"],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [10.0],
        }
    ).to_csv(detections, index=False)

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    assert abs(float(candidates.rows.loc[0, "time_s"]) - 3.5) < 1e-12


def test_camera_detections_json_backproject_to_world_candidates(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                        "translation_m": [1.0, 2.0, 3.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.json"
    detections.write_text(
        json.dumps(
            {
                "camera_detections": [
                    {
                        "sequence_id": "seq1",
                        "timestamp_ns": 2_000_000_000,
                        "source": "cam0",
                        "x1": 45.0,
                        "y1": 45.0,
                        "x2": 55.0,
                        "y2": 55.0,
                        "depth_m": 10.0,
                        "class": "Mavic3",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq1"
    assert abs(float(row["time_s"]) - 2.0) < 1e-12
    assert (row["x_m"], row["y_m"], row["z_m"]) == (1.0, 2.0, 13.0)
    assert row["class_name"] == "Mavic3"


def test_camera_detections_json_accepts_prediction_wrappers(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_predictions.json"
    detections.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "timestamp_s": 4.0,
                        "source": "cam0",
                        "bbox": [45.0, 45.0, 10.0, 10.0],
                        "depth": 8.0,
                        "score": 0.9,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert abs(float(row["time_s"]) - 4.0) < 1.0e-12
    assert abs(float(row["x_m"])) < 1.0e-9
    assert abs(float(row["y_m"])) < 1.0e-9
    assert abs(float(row["z_m"]) - 8.0) < 1.0e-9
    assert abs(float(row["confidence"]) - 0.9) < 1.0e-12


def test_camera_detections_json_accepts_detection2d_style_rows(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "detection2d_export.json"
    detections.write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "header": {
                            "stamp": {"sec": 5, "nanosec": 250_000_000},
                            "frame_id": "cam0",
                        },
                        "bbox": {
                            "center": {"position": {"x": 60.0, "y": 55.0}},
                            "size_x": 20.0,
                            "size_y": 10.0,
                        },
                        "depth_m": 10.0,
                        "results": [
                            {"hypothesis": {"class_id": "uav", "score": 0.8}},
                            {"hypothesis": {"class_id": "bird", "score": 0.2}},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert row["source"] == "cam0"
    assert row["class_name"] == "uav"
    assert abs(float(row["time_s"]) - 5.25) < 1.0e-12
    assert abs(float(row["x_m"]) - 1.0) < 1.0e-9
    assert abs(float(row["y_m"]) - 0.5) < 1.0e-9
    assert abs(float(row["z_m"]) - 10.0) < 1.0e-9
    assert abs(float(row["confidence"]) - 0.8) < 1.0e-12


def test_camera_detections_json_accepts_detection2d_center_depth(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "detection2d_export.json"
    detections.write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "header": {
                            "stamp": {"sec": 6, "nanosec": 500_000_000},
                            "frame_id": "cam0",
                        },
                        "bbox": {
                            "center": {"position": {"x": 50.0, "y": 50.0, "z": 7.0}},
                            "size_x": 12.0,
                            "size_y": 8.0,
                        },
                        "results": [{"hypothesis": {"class_id": "uav", "score": 0.75}}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert row["source"] == "cam0"
    assert abs(float(row["time_s"]) - 6.5) < 1.0e-12
    assert abs(float(row["x_m"])) < 1.0e-9
    assert abs(float(row["y_m"])) < 1.0e-9
    assert abs(float(row["z_m"]) - 7.0) < 1.0e-9
    assert abs(float(row["confidence"]) - 0.75) < 1.0e-12


def test_camera_detections_jsonl_backproject_to_world_candidates(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                        "translation_m": [1.0, 0.0, 0.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.jsonl"
    rows = [
        {
            "timestamp_s": 1.0,
            "source": "cam0",
            "bbox": [45.0, 45.0, 10.0, 10.0],
            "depth_m": 10.0,
        }
    ]
    detections.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert abs(float(row["time_s"]) - 1.0) < 1.0e-12
    assert abs(float(row["x_m"]) - 1.0) < 1.0e-9
    assert abs(float(row["z_m"]) - 10.0) < 1.0e-9


def test_camera_detections_gzipped_jsonl_backproject_to_world_candidates(
    tmp_path: Path,
) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.jsonl.gz"
    with gzip.open(detections, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp_s": 1.0,
                    "source": "cam0",
                    "bbox": [45.0, 45.0, 10.0, 10.0],
                    "depth_m": 10.0,
                }
            )
        )

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert abs(float(row["time_s"]) - 1.0) < 1.0e-12
    assert abs(float(row["z_m"]) - 10.0) < 1.0e-9


def test_camera_detections_json_accepts_coco_bbox_xywh(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.json"
    detections.write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "sequence_id": "seq1",
                        "time_s": 1.25,
                        "source": "cam0",
                        "bbox": [40.0, 40.0, 20.0, 20.0],
                        "depth_m": 10.0,
                        "score": 0.8,
                        "category": "Mavic3",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert row["sequence_id"] == "seq1"
    assert abs(float(row["time_s"]) - 1.25) < 1e-12
    assert (row["x_m"], row["y_m"], row["z_m"]) == (0.0, 0.0, 10.0)
    assert row["confidence"] == 0.8
    assert row["class_name"] == "Mavic3"


def test_camera_detections_csv_accepts_compact_bbox_xyxy_string(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.csv"
    detections.write_text(
        'sequence_id,time_s,source,bbox_xyxy,depth_m\nseq1,0.0,cam0,"[45, 45, 55, 55]",10.0\n',
        encoding="utf-8",
    )

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert (row["x_m"], row["y_m"], row["z_m"]) == (0.0, 0.0, 10.0)


def test_cli_accepts_explicit_camera_detection_json_file(tmp_path: Path) -> None:
    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.json"
    detections.write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "sequence_id": "default",
                        "time_s": 0.0,
                        "source": "cam0",
                        "u_px": 50.0,
                        "v_px": 50.0,
                        "depth_m": 5.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["default"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [5.0],
        }
    ).to_csv(truth, index=False)
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--camera-detections-file",
            str(detections),
            "--camera-calibration-file",
            str(calibration),
            "--truth-csv",
            str(truth),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output / "mmuad_estimates.csv")
    assert estimates["sequence_id"].tolist() == ["default"]
    metrics = json.loads((output / "mmuad_metrics.json").read_text(encoding="utf-8"))
    assert metrics["pooled"]["mean_3d_m"] == 0.0


def test_cli_accepts_explicit_camera_source_for_source_less_detection_file(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "detections.csv"
    pd.DataFrame(
        {
            "sequence_id": ["default"],
            "time_s": [0.0],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [5.0],
        }
    ).to_csv(detections, index=False)
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["default"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [5.0],
        }
    ).to_csv(truth, index=False)
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--camera-detections-file",
            str(detections),
            "--camera-calibration-file",
            str(calibration),
            "--camera-source",
            "cam0",
            "--truth-file",
            str(truth),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    assert (output / "mmuad_estimates.csv").exists()


def test_camera_detections_fill_blank_source_from_default_source(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import (
        load_camera_detections_csv_as_candidates,
        load_camera_models,
    )

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "detections.csv"
    pd.DataFrame(
        {
            "sequence_id": [np.nan, ""],
            "time_s": [0.0, 1.0],
            "source": [np.nan, ""],
            "u_px": [50.0, 50.0],
            "v_px": [50.0, 50.0],
            "depth_m": [5.0, 5.0],
        }
    ).to_csv(detections, index=False)

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
        default_source="cam0",
        sequence_id="seq_camera",
    )

    assert candidates.rows["source"].tolist() == ["cam0", "cam0"]
    assert candidates.rows["sequence_id"].tolist() == ["seq_camera", "seq_camera"]
    assert candidates.rows[["x_m", "y_m", "z_m"]].values.tolist() == [
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 5.0],
    ]


def test_cli_accepts_repeated_camera_calibration_files_with_folder_sources(
    tmp_path: Path,
) -> None:
    cam0 = tmp_path / "cam0"
    cam1 = tmp_path / "cam1"
    cam0.mkdir()
    cam1.mkdir()
    camera_info = {
        "width": 100,
        "height": 100,
        "k": [
            100.0,
            0.0,
            50.0,
            0.0,
            100.0,
            50.0,
            0.0,
            0.0,
            1.0,
        ],
    }
    (cam0 / "camera_info.json").write_text(json.dumps(camera_info), encoding="utf-8")
    cam1_info = dict(camera_info)
    cam1_info["translation_m"] = [10.0, 0.0, 0.0]
    (cam1 / "camera_info.json").write_text(json.dumps(cam1_info), encoding="utf-8")
    pd.DataFrame(
        {
            "sequence_id": ["default"],
            "time_s": [0.0],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [5.0],
        }
    ).to_csv(cam0 / "detections.csv", index=False)
    pd.DataFrame(
        {
            "sequence_id": ["default"],
            "time_s": [1.0],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [5.0],
        }
    ).to_csv(cam1 / "detections.csv", index=False)
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["default", "default"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [5.0, 5.0],
        }
    ).to_csv(truth, index=False)
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--camera-detections-csv",
            str(cam0 / "detections.csv"),
            "--camera-detections-csv",
            str(cam1 / "detections.csv"),
            "--camera-calibration-file",
            str(cam0 / "camera_info.json"),
            "--camera-calibration-file",
            str(cam1 / "camera_info.json"),
            "--truth-csv",
            str(truth),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    selected = pd.read_csv(output / "mmuad_selected_tracklets.csv")
    assert selected["source"].tolist() == ["cam0", "cam1"]
    assert selected["sequence_id"].tolist() == ["default", "default"]
    assert abs(float(selected.loc[0, "x_m"])) < 1.0e-9
    assert abs(float(selected.loc[1, "x_m"]) - 10.0) < 1.0e-9
    estimates = pd.read_csv(output / "mmuad_estimates.csv")
    assert estimates["source"].tolist() == ["cam0", "cam1"]
    assert estimates["sequence_id"].tolist() == ["default", "default"]
    assert abs(float(estimates.loc[0, "state_x_m"])) < 1.0e-9
    metrics = json.loads((output / "mmuad_metrics.json").read_text(encoding="utf-8"))
    assert float(metrics["pooled"]["mean_3d_m"]) < 1.0


def test_cli_accepts_explicit_radar_polar_json_file(tmp_path: Path) -> None:
    radar = tmp_path / "radar_polar.json"
    radar.write_text(
        json.dumps(
            {
                "radar_polar": [
                    {
                        "sequence_id": "default",
                        "time_s": 0.0,
                        "range_m": 10.0,
                        "azimuth_deg": 90.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["default"],
            "time_s": [0.0],
            "x_m": [10.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    ).to_csv(truth, index=False)
    output = tmp_path / "out"

    status = mmuad_cli_main(
        [
            "--radar-polar-file",
            str(radar),
            "--radar-azimuth-convention",
            "north-clockwise",
            "--truth-csv",
            str(truth),
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    estimates = pd.read_csv(output / "mmuad_estimates.csv")
    assert abs(float(estimates.loc[0, "state_x_m"]) - 10.0) < 1.0e-9
    metrics = json.loads((output / "mmuad_metrics.json").read_text(encoding="utf-8"))
    assert abs(float(metrics["pooled"]["mean_3d_m"])) < 1.0e-9


def test_sequence_export_applies_discovered_sensor_uncertainty_options(
    tmp_path: Path,
) -> None:
    seq = tmp_path / "seq_sensor_uncertainty"
    seq.mkdir()
    (seq / "calibration.json").write_text(
        json.dumps(
            {
                "sensors": {
                    "radar_polar": {},
                    "cam0": {},
                },
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": [10.0],
            "azimuth_deg": [0.0],
        }
    ).to_csv(seq / "radar_polar.csv", index=False)
    pd.DataFrame(
        {
            "time_s": [1.0],
            "source": ["cam0"],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [5.0],
        }
    ).to_csv(seq / "camera_detections.csv", index=False)

    candidates, _truth, _calibration = load_sequence_export(
        discover_sequence_paths(tmp_path)[0],
        apply_calibration=False,
        radar_polar_range_std_m=9.0,
        radar_polar_angle_std_deg=2.0,
        radar_polar_z_std_m=4.5,
        camera_std_xy_m=11.0,
        camera_std_z_m=22.0,
    )

    radar_row = candidates.rows.loc[candidates.rows["source"] == "radar_polar"].iloc[0]
    camera_row = candidates.rows.loc[candidates.rows["source"] == "cam0"].iloc[0]

    assert radar_row["std_xy_m"] == 9.0
    assert radar_row["std_z_m"] == 4.5
    assert camera_row["std_xy_m"] == 11.0
    assert camera_row["std_z_m"] == 22.0


def test_camera_detections_txt_backproject_to_world_candidates(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.txt"
    detections.write_text(
        "time source u_px v_px depth_m\n0 cam0 50 50 10\n",
        encoding="utf-8",
    )

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert row["source"] == "cam0"
    assert (row["x_m"], row["y_m"], row["z_m"]) == (0.0, 0.0, 10.0)


def test_camera_models_accept_opencv_matrix_intrinsics(tmp_path: Path) -> None:
    from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models

    calibration = tmp_path / "camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "cam0": {
                        "camera_matrix": {
                            "rows": 3,
                            "cols": 3,
                            "data": [
                                100.0,
                                0.0,
                                50.0,
                                0.0,
                                100.0,
                                50.0,
                                0.0,
                                0.0,
                                1.0,
                            ],
                        },
                        "T_camera_to_world": [
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            2.0,
                            0.0,
                            0.0,
                            1.0,
                            3.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    detections = tmp_path / "camera_detections.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [0.0],
            "source": ["cam0"],
            "u_px": [50.0],
            "v_px": [50.0],
            "depth_m": [10.0],
        }
    ).to_csv(detections, index=False)

    candidates = load_camera_detections_csv_as_candidates(
        detections,
        camera_models=load_camera_models(calibration),
    )

    row = candidates.rows.iloc[0]
    assert (row["x_m"], row["y_m"], row["z_m"]) == (1.0, 2.0, 13.0)


def test_infer_sequence_class_map_from_candidate_votes() -> None:
    from raft_uav.mmuad.classification import infer_sequence_class_map_from_candidates
    from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

    candidates = CandidateFrame(
        normalize_candidate_columns(
            pd.DataFrame(
                {
                    "sequence_id": ["seq1", "seq1", "seq2"],
                    "time_s": [0.0, 1.0, 0.0],
                    "source": ["camera", "camera", "radar"],
                    "x_m": [0.0, 1.0, 0.0],
                    "y_m": [0.0, 1.0, 0.0],
                    "z_m": [5.0, 5.0, 6.0],
                    "confidence": [0.4, 0.9, 0.8],
                    "class_name": ["Phantom4", "Mavic3", "uav"],
                }
            )
        )
    )

    mapping = infer_sequence_class_map_from_candidates(candidates, default_class="unknown")

    assert mapping["seq1"] == "Mavic3"
    assert mapping["seq2"] == "unknown"


def test_mmaud_results_empty_truth_still_writes_artifacts(tmp_path: Path) -> None:
    results = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "timestamp": [0.0],
            "x": [0.0],
            "y": [0.0],
            "z": [10.0],
            "uav_type": ["Mavic3"],
            "score": [1.0],
        }
    )
    truth = pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])

    evaluated = evaluate_mmaud_results(results, truth)

    assert evaluated["summary"]["count"] == 1
    assert evaluated["summary"]["matched_count"] == 0
    assert evaluated["summary"]["unmatched_count"] == 1
    assert evaluated["summary"]["pooled"]["count"] == 0
    assert evaluated["rows"].loc[0, "unmatched_reason"] == "empty_truth"

    paths = write_evaluation_artifacts(
        evaluated,
        summary_json=tmp_path / "eval.json",
        rows_csv=tmp_path / "eval_rows.csv",
    )

    assert Path(paths["evaluation_json"]).exists()
    assert Path(paths["evaluation_rows_csv"]).exists()
    json.loads(Path(paths["evaluation_json"]).read_text(encoding="utf-8"))
