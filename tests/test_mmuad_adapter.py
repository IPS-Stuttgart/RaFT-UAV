from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.io import load_candidate_csv, load_point_cloud_csv_as_candidates, load_truth_csv
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker, write_tracker_output


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

from raft_uav.mmuad.calibration import load_calibration_json, transform_candidate_frame
from raft_uav.mmuad.sequence import discover_sequence_paths, load_sequence_export
from raft_uav.mmuad.submission import estimates_to_submission_frame, write_submission_json


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
