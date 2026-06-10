from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from raft_uav.mmuad.calibration import load_calibration_json, transform_candidate_frame
from raft_uav.mmuad.io import (
    load_candidate_csv,
    load_point_cloud_csv_as_candidates,
    load_point_cloud_file_as_candidates,
    load_truth_csv,
    merge_candidate_frames,
)
from raft_uav.mmuad.mot import (
    MultiObjectTrackerConfig,
    compute_multi_object_metrics,
    run_mmuad_multi_object_tracker,
)
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame
from raft_uav.mmuad.sequence import discover_sequence_paths, load_sequence_export
from raft_uav.mmuad.splits import filter_sequences_by_split, load_split_manifest
from raft_uav.mmuad.submission import (
    compute_trajectory_metrics,
    estimates_to_submission_frame,
    write_submission_json,
    write_submission_zip,
)
from raft_uav.mmuad.tracker import (
    TrackerConfig,
    add_truth_errors,
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
