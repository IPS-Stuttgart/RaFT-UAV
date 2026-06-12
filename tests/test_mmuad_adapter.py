from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd

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
from raft_uav.mmuad.pointcloud2 import pointcloud2_to_candidates, pointcloud2_to_dataframe
from raft_uav.mmuad.rosbag_bridge import (
    inspect_rosbag,
    load_topic_map_exports,
    write_topic_map_template,
)
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame
from raft_uav.mmuad.sequence import discover_sequence_paths, load_sequence_export
from raft_uav.mmuad.splits import filter_sequences_by_split, load_split_manifest
from raft_uav.mmuad.submission import (
    compute_trajectory_metrics,
    estimates_to_mmaud_results_frame,
    estimates_to_submission_frame,
    inspect_submission_zip,
    load_sequence_class_map,
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


def test_sequence_class_map_accepts_csv_alias_columns(tmp_path: Path) -> None:
    class_map_csv = tmp_path / "classes.csv"
    class_map_csv.write_text("id,type\nseqA,Mavic3\nseqB,Phantom4\n", encoding="utf-8")

    mapping = load_sequence_class_map(class_map_csv)

    assert mapping == {"seqA": "Mavic3", "seqB": "Phantom4"}


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
    assert detailed["category_counts"]["topic_map_native"] == 1
    by_sequence = {
        row["sequence_id"]: row["missing_for_tracking_smoke"]
        for row in detailed["sequences"]
    }
    assert "truth" not in by_sequence["seq_exported_topic_map"]
    assert "candidate_or_point_cloud" not in by_sequence["seq_exported_topic_map"]
    assert "candidate_or_point_cloud" in by_sequence["seq_native_topic_map"]

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["topic_map_export"] == 1
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
    assert exported_summary["has_topic_map_export"] is True
    assert exported_summary["has_candidates_or_points"] is True
    assert exported_summary["has_truth_or_labels"] is True
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
    (seq / "classes.json").write_text(
        json.dumps({"seq_json_tables": "quadrotor"}),
        encoding="utf-8",
    )

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}
    assert by_name["candidates.json"]["category"] == "candidate"
    assert by_name["truth.json"]["category"] == "truth"
    assert by_name["classes.json"]["category"] == "class_label"
    assert detailed["sequences"][0]["missing_for_tracking_smoke"] == ["calibration"]

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["candidate_or_point_table"] == 1
    assert inventory["category_counts"]["truth_or_label"] == 1
    assert inventory["category_counts"]["class_or_label"] == 1
    sequence = inventory["sequence_candidates"][0]
    assert sequence["has_candidates_or_points"] is True
    assert sequence["has_truth_or_labels"] is True
    assert sequence["has_class_labels"] is True


def test_layout_inspectors_classify_mmuad_modality_folders(tmp_path: Path) -> None:
    seq = tmp_path / "seq_foldered"
    (seq / "livox_avia").mkdir(parents=True)
    (seq / "ground_truth").mkdir()
    (seq / "tracking_results").mkdir()
    (seq / "class").mkdir()
    np.save(seq / "livox_avia" / "20.0.npy", np.zeros((3, 3)))
    np.save(seq / "ground_truth" / "20.0.npy", np.array([0.0, 0.0, 1.0]))
    np.save(seq / "tracking_results" / "20.0.npy", np.array([0.0, 0.0, 1.0]))
    np.save(seq / "class" / "20.0.npy", np.array(2))

    detailed = inspect_sequence_root(tmp_path)
    by_name = {row["relative_path"]: row for row in detailed["files"]}

    assert by_name["livox_avia/20.0.npy"]["category"] == "point_cloud"
    assert by_name["ground_truth/20.0.npy"]["category"] == "truth"
    assert by_name["tracking_results/20.0.npy"]["category"] == "candidate"
    assert by_name["class/20.0.npy"]["category"] == "class_label"
    assert detailed["sequences"][0]["missing_for_tracking_smoke"] == ["calibration"]

    inventory = inspect_mmuad_layout(tmp_path)
    assert inventory["category_counts"]["truth_or_label"] == 1
    assert inventory["category_counts"]["class_or_label"] == 1
    assert inventory["category_counts"]["candidate_or_point_table"] == 2
    sequence = inventory["sequence_candidates"][0]
    assert sequence["has_truth_or_labels"] is True
    assert sequence["has_candidates_or_points"] is True
    assert sequence["has_class_labels"] is True


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


def test_ros2_metadata_inspection_and_topic_map_template(tmp_path: Path) -> None:
    bag = tmp_path / "bagdir"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  topics_with_message_count:",
                "    - topic_metadata:",
                "        name: /radar/points",
                "        type: sensor_msgs/msg/PointCloud2",
                "      message_count: 3",
                "    - topic_metadata:",
                "        name: /ground_truth",
                "        type: geometry_msgs/msg/PoseStamped",
                "      message_count: 3",
                "    - topic_metadata:",
                "        name: /detector/odom",
                "        type: nav_msgs/msg/Odometry",
                "      message_count: 3",
            ]
        ),
        encoding="utf-8",
    )
    report = inspect_rosbag(bag)
    assert report["kind"] == "ros2_bag_directory"
    assert len(report["topics"]) == 3
    template = write_topic_map_template(report, tmp_path / "topic_map_template.json")
    payload = json.loads(template.read_text(encoding="utf-8"))
    assert payload["schema"] == "raft-uav-mmuad-topic-map-v1"
    assert [entry["kind"] for entry in payload["exports"]] == [
        "pointcloud2_candidate",
        "pose_truth",
        "odometry_candidate",
    ]
    assert payload["exports"][0]["source"] == "radar_points"
    assert payload["exports"][1]["source"] is None


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
