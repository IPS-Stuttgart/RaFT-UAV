import json
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
    load_mmaud_results_file,
    validate_mmaud_results_frame,
)
from raft_uav.mmuad.submission import (
    OFFICIAL_UG2_RESULT_COLUMNS,
    estimates_to_official_mmaud_results_frame,
    estimates_to_mmaud_results_frame,
    estimates_to_submission_frame,
    parse_official_position_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
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


def test_results_frame_accepts_baseline_estimate_coordinate_columns():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0"],
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
        }
    )

    results = estimates_to_mmaud_results_frame(estimates, class_name="Mavic3")

    assert list(results["sequence_id"]) == ["seq0"]
    assert list(results["x"]) == [10.0]
    assert list(results["y"]) == [20.0]
    assert list(results["z"]) == [30.0]
    assert list(results["uav_type"]) == ["Mavic3"]


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


def test_submission_frame_accepts_baseline_estimate_coordinate_columns():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0", "seq0"],
            "time_s": [1.0, 0.5],
            "east_m": [10.0, 20.0],
            "north_m": [30.0, 40.0],
            "up_m": [50.0, 60.0],
            "output_track_id": ["track-a", None],
        }
    )

    frame = estimates_to_submission_frame(estimates, track_id="fallback")

    assert list(frame["sequence_id"]) == ["seq0", "seq0"]
    assert list(frame["time_s"]) == [0.5, 1.0]
    assert list(frame["track_id"]) == ["fallback", "track-a"]
    assert list(frame["x_m"]) == [20.0, 10.0]
    assert list(frame["y_m"]) == [40.0, 30.0]
    assert list(frame["z_m"]) == [60.0, 50.0]


def test_submission_frame_fills_blank_and_stringified_missing_output_track_ids():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0"] * 6,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_x_m": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "state_z_m": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "output_track_id": ["track-a", "", "   ", "nan", "None", "<NA>"],
        }
    )

    frame = estimates_to_submission_frame(estimates, track_id="fallback")

    assert list(frame["track_id"]) == [
        "track-a",
        "fallback",
        "fallback",
        "fallback",
        "fallback",
        "fallback",
    ]


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


def test_official_results_frame_uses_track5_columns_and_position_tuple():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq2", "seq1"],
            "time_s": [2.0, 1.0],
            "state_x_m": [10.0, 1.0],
            "state_y_m": [20.0, 2.0],
            "state_z_m": [30.0, 3.0],
            "class_name": ["1", "0"],
        }
    )

    results = estimates_to_official_mmaud_results_frame(estimates)

    assert list(results.columns) == list(OFFICIAL_UG2_RESULT_COLUMNS)
    assert results.to_dict("records") == [
        {
            "Sequence": "seq1",
            "Timestamp": 1.0,
            "Position": "(1,2,3)",
            "Classification": 0,
        },
        {
            "Sequence": "seq2",
            "Timestamp": 2.0,
            "Position": "(10,20,30)",
            "Classification": 1,
        },
    ]


def test_official_results_frame_requires_integer_classification():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [1.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            "class_name": ["Mavic3"],
        }
    )

    with pytest.raises(ValueError, match="Classification values must be integer ids"):
        estimates_to_official_mmaud_results_frame(estimates)


def test_official_results_csv_and_zip_are_readable_by_local_loader(tmp_path):
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
        }
    )

    csv_path = write_official_mmaud_results_csv(
        estimates,
        tmp_path / "mmaud_results.csv",
        class_map={"seqA": "2"},
    )
    zip_path = write_official_ug2_codabench_zip(
        estimates,
        tmp_path / "official_submission.zip",
        class_map={"seqA": "2"},
    )

    csv_frame = pd.read_csv(csv_path)
    assert list(csv_frame.columns) == list(OFFICIAL_UG2_RESULT_COLUMNS)
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
        zipped_frame = pd.read_csv(archive.open("mmaud_results.csv"))
    assert list(zipped_frame.columns) == list(OFFICIAL_UG2_RESULT_COLUMNS)

    loaded = load_mmaud_results_file(zip_path)
    assert loaded.rows["sequence_id"].tolist() == ["seqA", "seqA"]
    assert loaded.rows["uav_type"].tolist() == ["2", "2"]
    assert loaded.rows[["x", "y", "z"]].values.tolist() == [
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
    ]


def test_official_track5_results_loader_accepts_position_strings():
    frame = validate_mmaud_results_frame(
        pd.DataFrame(
            {
                "Sequence": ["seq1"],
                "Timestamp": [1706255054.386069],
                "Position": ["(1.5,2.5,3.5)"],
                "Classification": [3],
            }
        )
    )

    assert frame.loc[0, "sequence_id"] == "seq1"
    assert frame.loc[0, ["x", "y", "z"]].tolist() == [1.5, 2.5, 3.5]
    assert frame.loc[0, "uav_type"] == "3"


def test_official_position_parser_accepts_numpy_style_space_separated_strings():
    assert parse_official_position_cell("[1.5 2.5 3.5]") == (1.5, 2.5, 3.5)
    assert parse_official_position_cell("(1.5 2.5 3.5)") == (1.5, 2.5, 3.5)
    assert parse_official_position_cell("1.5 2.5 3.5") == (1.5, 2.5, 3.5)


def test_official_track5_results_loader_accepts_numpy_style_position_strings():
    frame = validate_mmaud_results_frame(
        pd.DataFrame(
            {
                "Sequence": ["seq1"],
                "Timestamp": [1706255054.386069],
                "Position": ["[1.5 2.5 3.5]"],
                "Classification": [3],
            }
        )
    )

    assert frame.loc[0, "sequence_id"] == "seq1"
    assert frame.loc[0, ["x", "y", "z"]].tolist() == [1.5, 2.5, 3.5]
    assert frame.loc[0, "uav_type"] == "3"


def test_official_track5_results_evaluator_reports_public_metric_aliases():
    results = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(2,0,0)"],
            "Classification": [1, 0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "uav_type": ["1", "1"],
        }
    )

    evaluated = evaluate_mmaud_results(results, truth)

    pooled = evaluated["summary"]["pooled"]
    assert pooled["pose_mse_loss_m2"] == 0.5
    assert pooled["mean_square_loss_m2"] == 0.5
    assert pooled["uav_type_accuracy"] == 0.5
    assert pooled["classification_accuracy"] == 0.5


def test_public_track5_metric_protocol_uses_truth_timestamp_denominator():
    results = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1", "seq1", "seq1"],
            "Timestamp": [0.0, 1.0, 1.0, 9.0],
            "Position": ["(0,0,0)", "(2,0,0)", "(3,0,0)", "(9,0,0)"],
            "Classification": [1, 0, 1, 1],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "uav_type": ["1", "1", "1"],
        }
    )

    evaluated = evaluate_mmaud_results(
        results,
        truth,
        metric_protocol="public-track5",
        timestamp_tolerance_s=0.0,
    )

    summary = evaluated["summary"]
    pooled = summary["pooled"]
    assert summary["metric_protocol"] == "public_track5_timestamp_aligned"
    assert summary["public_track5_metric"] is True
    assert summary["closed_codabench_evaluator"] is False
    assert summary["truth_count"] == 3
    assert summary["prediction_count"] == 4
    assert summary["matched_count"] == 2
    assert summary["missing_prediction_count"] == 1
    assert summary["duplicate_prediction_count"] == 1
    assert summary["extra_prediction_count"] == 1
    assert summary["all_truth_timestamps_matched"] is False
    assert summary["truth_coverage_fraction"] == 2 / 3
    assert pooled["mean_square_loss_m2"] == 0.5
    assert pooled["classification_accuracy"] == 0.5
    reasons = evaluated["rows"]["unmatched_reason"].fillna("").tolist()
    assert "missing_prediction" in reasons
    assert "duplicate_prediction" in reasons
    assert "extra_prediction" in reasons


def test_public_track5_metric_normalizes_integer_like_class_labels():
    results = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(1,0,0)"],
            "Classification": ["2.0", 1],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "class_id": [2.0, 1.0],
        }
    )

    evaluated = evaluate_mmaud_results(
        results,
        truth,
        metric_protocol="public-track5",
        timestamp_tolerance_s=0.0,
    )

    rows = evaluated["rows"]
    assert rows["predicted_uav_type"].tolist() == ["2", "1"]
    assert rows["truth_uav_type"].tolist() == ["2", "1"]
    assert evaluated["summary"]["pooled"]["classification_accuracy"] == 1.0


def test_official_track5_submission_validator_accepts_exact_zip_and_template(tmp_path):
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [10.0, 10.0],
            "class_id": [2, 2],
        }
    )
    zip_path = write_official_ug2_codabench_zip(
        estimates,
        tmp_path / "official.zip",
    )
    template = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    validation = validate_official_track5_submission(zip_path, template=template)

    assert validation.summary["valid"] is True
    assert validation.summary["contains_only_mmaud_results_csv"] is True
    assert validation.summary["row_count"] == 2
    assert validation.summary["template_checked"] is True
    assert validation.summary["missing_template_timestamp_count"] == 0
    assert validation.summary["extra_prediction_count"] == 0
    assert validation.summary["duplicate_prediction_count"] == 0
    assert validation.rows.loc[validation.rows["row_type"] == "prediction", "status"].tolist() == [
        "ok",
        "ok",
    ]


def test_official_track5_submission_validator_accepts_numpy_style_position_strings(
    tmp_path,
):
    zip_path = tmp_path / "official.zip"
    frame = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["[1.5 2.5 3.5]"],
            "Classification": [2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))

    validation = validate_official_track5_submission(zip_path)

    assert validation.summary["valid"] is True
    assert validation.summary["invalid_position_count"] == 0
    row = validation.rows.iloc[0]
    assert row["status"] == "ok"
    assert row[["x", "y", "z"]].tolist() == [1.5, 2.5, 3.5]


def test_official_track5_submission_validator_rejects_bad_leaderboard_package(tmp_path):
    zip_path = tmp_path / "bad_official.zip"
    frame = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1", "seq1", "seq1", "seq2"],
            "Timestamp": [0.0, 1.0, 1.0, 9.0, 0.0],
            "Position": ["(0,0,0)", "(1,0,0)", "(1,0,0)", "(9,0,0)", "(0,0,0)"],
            "Classification": [2, 2, 2, 2, "not-an-int"],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))
        archive.writestr("README.txt", "extra files are not allowed")
    template = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
        }
    )

    validation = validate_official_track5_submission(
        zip_path,
        template=template,
        timestamp_tolerance_s=0.0,
    )

    assert validation.summary["valid"] is False
    assert validation.summary["contains_only_mmaud_results_csv"] is False
    assert "official Track 5 ZIP must contain only mmaud_results.csv" in validation.summary["errors"]
    assert validation.summary["invalid_classification_count"] == 1
    assert validation.summary["duplicate_prediction_count"] == 1
    assert validation.summary["missing_template_timestamp_count"] == 1
    assert validation.summary["extra_prediction_count"] == 1
    statuses = set(validation.rows["status"])
    assert "invalid_classification" in statuses
    assert "duplicate_prediction" in statuses
    assert "missing_template_timestamp" in statuses
    assert "extra_prediction" in statuses
