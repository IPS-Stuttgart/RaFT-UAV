from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.classification import (
    classify_sequences_from_features,
    infer_sequence_class_map_from_candidates,
    load_sequence_class_labels,
    sequence_features_from_rows,
    write_sequence_classification_result,
)
from raft_uav.mmuad.classification_cli import main as sequence_classifier_main
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


def _candidate_frame(rows: list[dict[str, object]]) -> CandidateFrame:
    return CandidateFrame(normalize_candidate_columns(pd.DataFrame.from_records(rows)))


def test_inferred_class_map_keeps_sequences_filtered_by_confidence() -> None:
    candidates = _candidate_frame(
        [
            {
                "sequence_id": "seq_low_confidence",
                "time_s": 0.0,
                "source": "camera",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "quadrotor",
                "confidence": 0.1,
            },
            {
                "sequence_id": "seq_confident",
                "time_s": 0.0,
                "source": "camera",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "hexrotor",
                "confidence": 0.9,
            },
        ]
    )

    class_map = infer_sequence_class_map_from_candidates(
        candidates,
        min_confidence=0.5,
        default_class="unknown",
    )

    assert class_map == {
        "seq_confident": "hexrotor",
        "seq_low_confidence": "unknown",
    }


def test_sequence_features_use_state_positions_and_velocity() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [10.0, 11.0, 12.0],
            "source": ["lidar_360", "lidar_360", "radar_enhance_pcl"],
            "state_x_m": [0.0, 3.0, 6.0],
            "state_y_m": [0.0, 4.0, 8.0],
            "state_z_m": [1.0, 2.0, 3.0],
            "v_x_mps": [3.0, 3.0, 3.0],
            "v_y_mps": [4.0, 4.0, 4.0],
            "v_z_mps": [0.5, 0.5, 0.5],
        }
    )

    features = sequence_features_from_rows(rows).set_index("sequence_id")

    assert features.loc["seqA", "row_count"] == 3
    assert features.loc["seqA", "duration_s"] == 2.0
    assert features.loc["seqA", "trajectory_displacement_2d_m"] == 10.0
    assert features.loc["seqA", "speed_xy_mps_mean"] == 5.0
    assert features.loc["seqA", "source_count_lidar_360"] == 2


def test_nearest_neighbor_sequence_classifier_writes_class_map(tmp_path) -> None:
    train_rows = pd.DataFrame(
        {
            "sequence_id": ["seq0", "seq0", "seq1", "seq1", "seq2", "seq2"],
            "time_s": [0, 1, 0, 1, 0, 1],
            "x_m": [0.0, 0.1, 10.0, 10.1, 20.0, 20.1],
            "y_m": [0.0, 0.1, 10.0, 10.1, 20.0, 20.1],
            "z_m": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        }
    )
    predict_rows = pd.DataFrame(
        {
            "sequence_id": ["seq_target", "seq_target"],
            "time_s": [0, 1],
            "x_m": [9.9, 10.2],
            "y_m": [9.9, 10.2],
            "z_m": [2.0, 2.0],
        }
    )

    result = classify_sequences_from_features(
        train_features=sequence_features_from_rows(train_rows),
        train_labels={"seq0": "0", "seq1": "1", "seq2": "2"},
        predict_features=sequence_features_from_rows(predict_rows),
        method="nearest-neighbor",
    )
    paths = write_sequence_classification_result(
        result,
        output_class_map=tmp_path / "class_map.csv",
        predictions_csv=tmp_path / "predictions.csv",
        metrics_json=tmp_path / "metrics.json",
    )

    assert result.predictions["predicted_class"].tolist() == ["1"]
    assert pd.read_csv(paths["class_map_csv"]).to_dict("records") == [
        {"sequence_id": "seq_target", "uav_type": 1}
    ]
    assert "sequence_accuracy" not in result.metrics


def test_sequence_class_labels_read_official_truth(tmp_path) -> None:
    truth_path = tmp_path / "official_truth.csv"
    pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqB"],
            "Timestamp": [1.0, 2.0, 1.0],
            "Position": ["(0,0,0)", "(1,1,1)", "(2,2,2)"],
            "Classification": [2, 2, 3],
        }
    ).to_csv(truth_path, index=False)

    assert load_sequence_class_labels(truth_path) == {"seqA": "2", "seqB": "3"}


def test_sequence_classifier_cli_writes_outputs(tmp_path) -> None:
    train_path = tmp_path / "train_features.csv"
    predict_path = tmp_path / "predict_features.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 5.0],
            "y_m": [0.0, 5.0],
            "z_m": [1.0, 2.0],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqTarget"],
            "time_s": [0.0],
            "x_m": [5.2],
            "y_m": [5.1],
            "z_m": [2.0],
        }
    ).to_csv(predict_path, index=False)
    pd.DataFrame({"sequence_id": ["seqA", "seqB"], "uav_type": [0, 1]}).to_csv(
        labels_path, index=False
    )

    assert (
        sequence_classifier_main(
            [
                "--train-feature-table",
                str(train_path),
                "--predict-feature-table",
                str(predict_path),
                "--train-labels",
                str(labels_path),
                "--output-class-map",
                str(tmp_path / "out_class_map.csv"),
                "--predictions-csv",
                str(tmp_path / "predictions.csv"),
                "--metrics-json",
                str(tmp_path / "metrics.json"),
            ]
        )
        == 0
    )
    assert pd.read_csv(tmp_path / "out_class_map.csv")["uav_type"].tolist() == [1]
