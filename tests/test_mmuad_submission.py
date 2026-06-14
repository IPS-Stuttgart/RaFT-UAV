import json
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
    load_evaluation_truth_file,
    load_mmaud_results_file,
    validate_mmaud_results_frame,
)
from raft_uav.mmuad.submission import (
    OFFICIAL_UG2_RESULT_COLUMNS,
    estimates_to_official_mmaud_results_frame,
    estimates_to_mmaud_results_frame,
    estimates_to_submission_frame,
    load_official_track5_template_file,
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


def test_official_results_frame_rejects_boolean_classification():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [1.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            "class_id": [True],
        }
    )

    with pytest.raises(ValueError, match="not booleans"):
        estimates_to_official_mmaud_results_frame(estimates)


def test_official_results_frame_rejects_nonfinite_rows_by_default():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, float("nan")],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
            "class_id": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="non-finite Timestamp/Position"):
        estimates_to_official_mmaud_results_frame(estimates)


def test_official_results_frame_can_drop_nonfinite_rows_for_diagnostic_exports():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, float("nan")],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
            "class_id": [1, 1],
        }
    )

    results = estimates_to_official_mmaud_results_frame(
        estimates,
        invalid_row_policy="drop",
    )

    assert results.to_dict("records") == [
        {
            "Sequence": "seq1",
            "Timestamp": 0.0,
            "Position": "(1,3,5)",
            "Classification": 1,
        }
    ]


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


def test_official_track5_results_loader_rejects_blank_sequence():
    with pytest.raises(ValueError, match="Sequence values must be nonblank"):
        validate_mmaud_results_frame(
            pd.DataFrame(
                {
                    "Sequence": ["   "],
                    "Timestamp": [0.0],
                    "Position": ["(1.5,2.5,3.5)"],
                    "Classification": [3],
                }
            )
        )


def test_official_track5_results_loader_rejects_nonfinite_timestamp():
    with pytest.raises(ValueError, match="Timestamp values must be finite numbers"):
        validate_mmaud_results_frame(
            pd.DataFrame(
                {
                    "Sequence": ["seq1"],
                    "Timestamp": [float("nan")],
                    "Position": ["(1.5,2.5,3.5)"],
                    "Classification": [3],
                }
            )
        )


def test_official_track5_results_loader_rejects_noninteger_classification():
    with pytest.raises(ValueError, match="Classification values must be integer ids"):
        validate_mmaud_results_frame(
            pd.DataFrame(
                {
                    "Sequence": ["seq1"],
                    "Timestamp": [0.0],
                    "Position": ["(1.5,2.5,3.5)"],
                    "Classification": ["Mavic3"],
                }
            )
        )


def test_official_position_parser_accepts_numpy_style_space_separated_strings():
    assert parse_official_position_cell("[1.5 2.5 3.5]") == (1.5, 2.5, 3.5)
    assert parse_official_position_cell("(1.5 2.5 3.5)") == (1.5, 2.5, 3.5)
    assert parse_official_position_cell("1.5 2.5 3.5") == (1.5, 2.5, 3.5)


def test_official_position_parser_accepts_numpy_array_repr_strings():
    assert parse_official_position_cell("array([1.5, 2.5, 3.5])") == (
        1.5,
        2.5,
        3.5,
    )
    assert parse_official_position_cell("np.array([1.5 2.5 3.5])") == (
        1.5,
        2.5,
        3.5,
    )
    assert parse_official_position_cell(
        "numpy.array([1.5, 2.5, 3.5], dtype=float32)"
    ) == (1.5, 2.5, 3.5)


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


def test_official_track5_results_loader_accepts_numpy_array_position_repr():
    frame = validate_mmaud_results_frame(
        pd.DataFrame(
            {
                "Sequence": ["seq1"],
                "Timestamp": [1706255054.386069],
                "Position": ["array([1.5, 2.5, 3.5])"],
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


def test_public_track5_metric_accepts_official_truth_frame():
    results = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(2,0,0)"],
            "Classification": [2, 1],
        }
    )
    truth = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(1,0,0)"],
            "Classification": [2, 2],
        }
    )

    evaluated = evaluate_mmaud_results(
        results,
        truth,
        metric_protocol="public-track5",
        timestamp_tolerance_s=0.0,
    )

    rows = evaluated["rows"]
    pooled = evaluated["summary"]["pooled"]
    assert rows["truth_x_m"].tolist() == [0.0, 1.0]
    assert rows["truth_uav_type"].tolist() == ["2", "2"]
    assert pooled["mean_square_loss_m2"] == 0.5
    assert pooled["classification_accuracy"] == 0.5
    assert evaluated["summary"]["leaderboard_ready"] is True


def test_evaluation_truth_file_loader_accepts_official_truth_zip(tmp_path):
    zip_path = tmp_path / "official_truth.zip"
    truth = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [1.0],
            "Position": ["array([1.0, 2.0, 3.0])"],
            "Classification": [4],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", truth.to_csv(index=False))

    loaded = load_evaluation_truth_file(zip_path)

    assert loaded.rows.loc[0, "sequence_id"] == "seq1"
    assert loaded.rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]
    assert loaded.rows.loc[0, "class_name"] == "4"


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
    assert summary["leaderboard_ready"] is False
    assert summary["score_valid_for_leaderboard"] is False
    assert set(summary["leaderboard_blocking_reasons"]) == {
        "not_all_truth_timestamps_matched",
        "missing_predictions",
        "extra_predictions",
        "duplicate_predictions",
    }
    assert summary["sequences"]["seq1"]["leaderboard_ready"] is False
    assert pooled["mean_square_loss_m2"] == 0.5
    assert pooled["classification_accuracy"] == 0.5
    reasons = evaluated["rows"]["unmatched_reason"].fillna("").tolist()
    assert "missing_prediction" in reasons
    assert "duplicate_prediction" in reasons
    assert "extra_prediction" in reasons


def test_public_track5_metric_reports_extra_only_sequence_summary():
    results = pd.DataFrame(
        {
            "Sequence": ["seq1", "seq_extra"],
            "Timestamp": [0.0, 0.0],
            "Position": ["(0,0,0)", "(9,0,0)"],
            "Classification": [1, 1],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "uav_type": ["1"],
        }
    )

    evaluated = evaluate_mmaud_results(
        results,
        truth,
        metric_protocol="public-track5",
        timestamp_tolerance_s=0.0,
    )

    summary = evaluated["summary"]
    assert summary["matched_count"] == 1
    assert summary["extra_prediction_count"] == 1
    assert summary["leaderboard_ready"] is False
    assert set(summary["sequences"]) == {"seq1", "seq_extra"}
    assert summary["sequences"]["seq1"]["leaderboard_ready"] is True
    extra_summary = summary["sequences"]["seq_extra"]
    assert extra_summary["truth_count"] == 0
    assert extra_summary["prediction_count"] == 1
    assert extra_summary["matched_count"] == 0
    assert extra_summary["extra_prediction_count"] == 1
    assert extra_summary["leaderboard_ready"] is False
    assert extra_summary["leaderboard_blocking_reasons"] == [
        "no_truth_timestamps",
        "extra_predictions",
    ]


def test_public_track5_metric_rejects_empty_truth_template():
    results = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )
    truth = pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])

    evaluated = evaluate_mmaud_results(
        results,
        truth,
        metric_protocol="public-track5",
        timestamp_tolerance_s=0.0,
    )

    summary = evaluated["summary"]
    assert summary["metric_protocol"] == "public_track5_timestamp_aligned"
    assert summary["public_track5_metric"] is True
    assert summary["closed_codabench_evaluator"] is False
    assert summary["truth_count"] == 0
    assert summary["prediction_count"] == 1
    assert summary["matched_count"] == 0
    assert summary["missing_prediction_count"] == 0
    assert summary["extra_prediction_count"] == 1
    assert summary["duplicate_prediction_count"] == 0
    assert summary["truth_coverage_fraction"] == 0.0
    assert summary["all_truth_timestamps_matched"] is False
    assert summary["leaderboard_ready"] is False
    assert summary["score_valid_for_leaderboard"] is False
    assert summary["leaderboard_blocking_reasons"] == [
        "no_truth_timestamps",
        "extra_predictions",
    ]
    assert summary["sequences"]["seq1"]["leaderboard_blocking_reasons"] == [
        "no_truth_timestamps",
        "extra_predictions",
    ]
    assert evaluated["rows"].loc[0, "unmatched_reason"] == "extra_prediction"


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
    assert evaluated["summary"]["leaderboard_ready"] is True
    assert evaluated["summary"]["score_valid_for_leaderboard"] is True
    assert evaluated["summary"]["leaderboard_blocking_reasons"] == []
    assert evaluated["summary"]["pooled"]["classification_accuracy"] == 1.0


def test_public_track5_metric_requires_one_prediction_per_truth_timestamp():
    results = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.02],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 0.04],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "uav_type": ["1", "1"],
        }
    )

    evaluated = evaluate_mmaud_results(
        results,
        truth,
        metric_protocol="public-track5",
        timestamp_tolerance_s=0.05,
    )

    summary = evaluated["summary"]
    reasons = evaluated["rows"]["unmatched_reason"].fillna("").tolist()
    assert summary["truth_count"] == 2
    assert summary["prediction_count"] == 1
    assert summary["matched_count"] == 1
    assert summary["missing_prediction_count"] == 1
    assert summary["truth_coverage_fraction"] == 0.5
    assert summary["leaderboard_ready"] is False
    assert summary["score_valid_for_leaderboard"] is False
    assert "missing_predictions" in summary["leaderboard_blocking_reasons"]
    assert "missing_prediction" in reasons


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
    assert validation.summary["score_valid_for_leaderboard"] is True
    assert validation.summary["leaderboard_ready"] is True
    assert validation.summary["codabench_upload_ready"] is True
    assert validation.summary["leaderboard_blocking_reasons"] == []
    assert validation.rows.loc[validation.rows["row_type"] == "prediction", "status"].tolist() == [
        "ok",
        "ok",
    ]


def test_official_track5_submission_validator_requires_template_for_leaderboard_ready(
    tmp_path,
):
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [0.0],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [10.0],
            "class_id": [2],
        }
    )
    zip_path = write_official_ug2_codabench_zip(
        estimates,
        tmp_path / "official.zip",
    )

    validation = validate_official_track5_submission(zip_path)

    assert validation.summary["valid"] is True
    assert validation.summary["template_checked"] is False
    assert validation.summary["score_valid_for_leaderboard"] is False
    assert validation.summary["leaderboard_ready"] is False
    assert validation.summary["codabench_upload_ready"] is False
    assert validation.summary["leaderboard_blocking_reasons"] == [
        "timestamp_template_not_checked"
    ]


def test_official_track5_submission_validator_rejects_empty_template_for_leaderboard_ready(
    tmp_path,
):
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "time_s": [0.0],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [10.0],
            "class_id": [2],
        }
    )
    zip_path = write_official_ug2_codabench_zip(
        estimates,
        tmp_path / "official.zip",
    )
    template = pd.DataFrame(columns=["sequence_id", "time_s"])

    validation = validate_official_track5_submission(zip_path, template=template)

    assert validation.summary["valid"] is True
    assert validation.summary["template_checked"] is True
    assert validation.summary["template_timestamp_count"] == 0
    assert validation.summary["score_valid_for_leaderboard"] is False
    assert validation.summary["leaderboard_ready"] is False
    assert validation.summary["codabench_upload_ready"] is False
    assert validation.summary["leaderboard_blocking_reasons"] == [
        "no_template_timestamps"
    ]


def test_official_track5_submission_validator_requires_one_prediction_per_template_timestamp(
    tmp_path,
):
    zip_path = tmp_path / "one_prediction_two_template_times.zip"
    frame = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.02],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))
    template = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 0.04],
        }
    )

    validation = validate_official_track5_submission(
        zip_path,
        template=template,
        timestamp_tolerance_s=0.05,
    )

    assert validation.summary["valid"] is False
    assert validation.summary["template_timestamp_count"] == 2
    assert validation.summary["missing_template_timestamp_count"] == 1
    assert validation.summary["extra_prediction_count"] == 0
    statuses = validation.rows.loc[validation.rows["row_type"] == "template", "status"].tolist()
    assert statuses == ["covered_template_timestamp", "missing_template_timestamp"]


def test_official_track5_submission_validator_reports_sequence_readiness(
    tmp_path,
):
    zip_path = tmp_path / "sequence_readiness.zip"
    frame = pd.DataFrame(
        {
            "Sequence": ["seq_ok", "seq_extra"],
            "Timestamp": [0.0, 0.0],
            "Position": ["(0,0,0)", "(9,0,0)"],
            "Classification": [1, 1],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))
    template = pd.DataFrame(
        {
            "sequence_id": ["seq_ok", "seq_missing"],
            "time_s": [0.0, 0.0],
        }
    )

    validation = validate_official_track5_submission(
        zip_path,
        template=template,
        timestamp_tolerance_s=0.0,
    )

    summary = validation.summary
    assert summary["valid"] is False
    assert summary["missing_template_timestamp_count"] == 1
    assert summary["extra_prediction_count"] == 1
    assert set(summary["sequences"]) == {"seq_ok", "seq_missing", "seq_extra"}
    assert summary["sequences"]["seq_ok"]["leaderboard_ready"] is True
    assert summary["sequences"]["seq_ok"]["template_coverage_fraction"] == 1.0
    missing_summary = summary["sequences"]["seq_missing"]
    assert missing_summary["template_timestamp_count"] == 1
    assert missing_summary["prediction_count"] == 0
    assert missing_summary["missing_template_timestamp_count"] == 1
    assert missing_summary["leaderboard_blocking_reasons"] == [
        "official_missing_template_timestamps"
    ]
    extra_summary = summary["sequences"]["seq_extra"]
    assert extra_summary["template_timestamp_count"] == 0
    assert extra_summary["prediction_count"] == 1
    assert extra_summary["extra_prediction_count"] == 1
    assert extra_summary["leaderboard_blocking_reasons"] == [
        "no_template_timestamps",
        "official_extra_predictions",
    ]


def test_official_track5_template_loader_accepts_csv_and_zip(tmp_path):
    template_frame = pd.DataFrame(
        {
            "Sequence": ["seq2", "seq1", "seq1"],
            "Timestamp": [2.0, 1.0, 1.0],
            "Position": ["(0,0,0)", "(1,0,0)", "(1,0,0)"],
            "Classification": [2, 1, 1],
        }
    )
    csv_path = tmp_path / "official_template.csv"
    zip_path = tmp_path / "official_template.zip"
    template_frame.to_csv(csv_path, index=False)
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", template_frame.to_csv(index=False))

    csv_template = load_official_track5_template_file(csv_path)
    zip_template = load_official_track5_template_file(zip_path)

    assert csv_template.to_dict("records") == [
        {"sequence_id": "seq1", "time_s": 1.0},
        {"sequence_id": "seq2", "time_s": 2.0},
    ]
    assert zip_template.to_dict("records") == csv_template.to_dict("records")


def test_official_track5_submission_validator_rejects_missing_like_sequences(tmp_path):
    zip_path = tmp_path / "invalid_sequences.zip"
    frame = pd.DataFrame(
        {
            "Sequence": [float("nan"), "None", "<NA>", "seq1"],
            "Timestamp": [0.0, 1.0, 2.0, 3.0],
            "Position": ["(0,0,0)", "(1,0,0)", "(2,0,0)", "(3,0,0)"],
            "Classification": [2, 2, 2, 2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))

    validation = validate_official_track5_submission(zip_path)

    assert validation.summary["valid"] is False
    assert validation.summary["invalid_sequence_count"] == 3
    assert validation.rows["status"].tolist() == [
        "invalid_sequence",
        "invalid_sequence",
        "invalid_sequence",
        "ok",
    ]


def test_official_track5_submission_template_ignores_missing_like_sequences(tmp_path):
    zip_path = tmp_path / "official.zip"
    frame = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))
    template = pd.DataFrame(
        {
            "sequence_id": [float("nan"), "None", "<NA>", "seq1"],
            "time_s": [1.0, 2.0, 3.0, 0.0],
        }
    )

    validation = validate_official_track5_submission(zip_path, template=template)

    assert validation.summary["valid"] is True
    assert validation.summary["template_timestamp_count"] == 1
    assert validation.summary["missing_template_timestamp_count"] == 0
    assert validation.summary["extra_prediction_count"] == 0


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


def test_official_track5_submission_validator_accepts_numpy_array_position_repr(
    tmp_path,
):
    zip_path = tmp_path / "official.zip"
    frame = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["array([1.5, 2.5, 3.5], dtype=float64)"],
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


def test_official_track5_submission_validator_rejects_boolean_classification(tmp_path):
    zip_path = tmp_path / "bad_bool_classification.zip"
    frame = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [True],
        }
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))

    validation = validate_official_track5_submission(zip_path)

    assert validation.summary["valid"] is False
    assert validation.summary["invalid_classification_count"] == 1
    assert validation.rows.loc[0, "status"] == "invalid_classification"
    assert "not booleans" in validation.rows.loc[0, "reason"]
