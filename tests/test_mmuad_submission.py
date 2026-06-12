import json
from zipfile import ZipFile

import pandas as pd

from raft_uav.mmuad.submission import (
    estimates_to_mmaud_results_frame,
    estimates_to_submission_frame,
    write_submission_json,
    write_submission_zip,
)


def test_results_frame_applies_default_sequence_mapping_without_sequence_column():
    estimates = pd.DataFrame(
        {
            "time_s": [1.0, 0.5],
            "state_x_m": [10.0, 20.0],
            "state_y_m": [30.0, 40.0],
            "state_z_m": [50.0, 60.0],
        }
    )

    results = estimates_to_mmaud_results_frame(
        estimates,
        class_name="fallback",
        class_map={"default": "mapped"},
    )

    assert list(results["sequence_id"]) == ["default", "default"]
    assert list(results["uav_type"]) == ["mapped", "mapped"]
    assert list(results["timestamp"]) == [0.5, 1.0]


def test_submission_frame_fills_missing_and_blank_sequence_ids():
    estimates = pd.DataFrame(
        {
            "sequence_id": [None, ""],
            "time_s": [1.0, 0.5],
            "state_x_m": [10.0, 20.0],
            "state_y_m": [30.0, 40.0],
            "state_z_m": [50.0, 60.0],
        }
    )

    frame = estimates_to_submission_frame(estimates, track_id="track0")

    assert list(frame["sequence_id"]) == ["default", "default"]
    assert list(frame["time_s"]) == [0.5, 1.0]
    assert list(frame["track_id"]) == ["track0", "track0"]


def test_submission_json_and_zip_keep_default_sequence_rows(tmp_path):
    estimates = pd.DataFrame(
        {
            "sequence_id": [None, ""],
            "time_s": [0.0, 1.0],
            "state_x_m": [10.0, 20.0],
            "state_y_m": [30.0, 40.0],
            "state_z_m": [50.0, 60.0],
        }
    )

    json_path = write_submission_json(estimates, tmp_path / "submission.json")
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert set(json_payload["sequences"]) == {"default"}
    assert len(json_payload["sequences"]["default"]) == 2

    zip_path = write_submission_zip(estimates, tmp_path / "submission.zip")
    with ZipFile(zip_path) as archive:
        zip_payload = json.loads(archive.read("submission.json").decode("utf-8"))

    assert set(zip_payload["sequences"]) == {"default"}
    assert len(zip_payload["sequences"]["default"]) == 2
