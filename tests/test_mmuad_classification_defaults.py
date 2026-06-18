from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.classification import (
    classify_sequences_from_features,
    infer_sequence_class_map_from_candidates,
    load_sequence_class_labels,
    sequence_features_from_files,
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


def test_sequence_features_include_sensor_point_cluster_and_empty_radar_stats() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "source": ["lidar_360", "lidar_360", "radar_enhance_pcl", "radar_enhance_pcl"],
            "x_m": [3.0, 4.0, None, 6.0],
            "y_m": [4.0, 3.0, None, 8.0],
            "z_m": [10.0, 12.0, None, 15.0],
            "cluster_point_count": [10, 20, 0, 5],
            "cluster_extent_x_m": [0.5, 0.7, None, 1.0],
            "cluster_extent_y_m": [0.4, 0.5, None, 1.5],
            "cluster_extent_z_m": [0.3, 0.4, None, 0.8],
            "cluster_extent_3d_m": [0.7, 0.9, None, 2.0],
        }
    )

    features = sequence_features_from_rows(rows).set_index("sequence_id")

    assert features.loc["seqA", "cluster_point_count_mean"] == 8.75
    assert features.loc["seqA", "source_lidar_360_cluster_point_count_mean"] == 15.0
    assert features.loc["seqA", "source_lidar_360_cluster_point_count_p95"] == 19.5
    assert features.loc["seqA", "source_radar_enhance_pcl_cluster_point_count_mean"] == 2.5
    assert features.loc["seqA", "cluster_extent_3d_m_mean"] == 1.2
    assert features.loc["seqA", "source_radar_enhance_pcl_range_3d_m_mean"] == (6.0**2 + 8.0**2 + 15.0**2) ** 0.5
    assert features.loc["seqA", "radar_frame_count"] == 2
    assert features.loc["seqA", "radar_empty_frame_count"] == 1
    assert features.loc["seqA", "radar_empty_frame_fraction"] == 0.5


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


def test_sequence_feature_reader_accepts_saved_sequence_feature_tables(tmp_path) -> None:
    feature_table = tmp_path / "sequence_features.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "cluster_point_count_mean": [10.0, 40.0],
            "cluster_extent_3d_m_mean": [0.5, 2.0],
        }
    ).to_csv(feature_table, index=False)

    features = sequence_features_from_files([feature_table])

    assert features["sequence_id"].tolist() == ["seqA", "seqB"]
    assert features["cluster_point_count_mean"].tolist() == [10.0, 40.0]
    assert features["cluster_extent_3d_m_mean"].tolist() == [0.5, 2.0]


@pytest.mark.parametrize("method", ["random-forest", "hist-gradient-boosting"])
def test_sklearn_sequence_classifiers_predict_one_label_per_sequence(method: str) -> None:
    pytest.importorskip("sklearn")
    train_features = pd.DataFrame(
        {
            "sequence_id": ["c0a", "c0b", "c1a", "c1b", "c2a", "c2b"],
            "cluster_point_count_mean": [8.0, 9.0, 30.0, 31.0, 80.0, 82.0],
            "cluster_extent_3d_m_mean": [0.4, 0.45, 1.2, 1.1, 2.4, 2.6],
            "diff_speed_mps_mean": [2.0, 2.2, 6.0, 6.5, 12.0, 12.5],
        }
    )
    predict_features = pd.DataFrame(
        {
            "sequence_id": ["target"],
            "cluster_point_count_mean": [81.0],
            "cluster_extent_3d_m_mean": [2.5],
            "diff_speed_mps_mean": [12.2],
        }
    )

    result = classify_sequences_from_features(
        train_features=train_features,
        train_labels={
            "c0a": "0",
            "c0b": "0",
            "c1a": "1",
            "c1b": "1",
            "c2a": "2",
            "c2b": "2",
        },
        predict_features=predict_features,
        method=method,
    )

    assert result.predictions["sequence_id"].tolist() == ["target"]
    assert result.predictions["predicted_class"].tolist() == ["2"]
    assert result.metrics["method"] == method
    assert "cluster_point_count_mean" in result.metrics["feature_columns"]


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
