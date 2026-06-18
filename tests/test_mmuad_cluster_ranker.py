from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.cluster_ranker import (
    build_cluster_feature_table,
    load_cluster_ranker_model,
    main as cluster_ranker_main,
    merge_cross_sensor_candidate_clusters,
    predict_cluster_scores,
    save_cluster_ranker_model,
    score_cluster_candidates,
    train_cluster_ranker,
)
from raft_uav.mmuad.cli import main as mmuad_cli_main
from raft_uav.mmuad.io import point_rows_to_candidates
from raft_uav.mmuad.schema import CandidateFrame


def _point_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 8,
            "time_s": [0.0] * 8,
            "x_m": [0.0, 0.1, 0.0, 0.1, 10.0, 10.1, 10.0, 10.1],
            "y_m": [0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
            "z_m": [1.0, 1.1, 1.0, 1.1, 4.0, 4.1, 4.0, 4.1],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.05],
            "y_m": [0.05],
            "z_m": [1.05],
        }
    )


def test_point_cloud_candidates_include_cluster_geometry_features() -> None:
    candidates = point_rows_to_candidates(_point_rows(), voxel_size_m=0.5, min_points=3)

    assert len(candidates.rows) == 2
    assert "cluster_point_count" in candidates.rows.columns
    assert candidates.rows["cluster_point_count"].tolist() == [4, 4]
    assert candidates.rows["cluster_density_points_per_m3"].notna().all()
    assert candidates.rows["cluster_range_3d_m"].iloc[0] < candidates.rows["cluster_range_3d_m"].iloc[1]


def test_cluster_ranker_labels_and_scores_good_cluster_higher() -> None:
    candidates = point_rows_to_candidates(_point_rows(), voxel_size_m=0.5, min_points=3)
    features = build_cluster_feature_table(
        candidates,
        truth=_truth_rows(),
        good_threshold_m=1.0,
        max_truth_time_delta_s=0.1,
    )

    assert features["good_cluster"].tolist() == [True, False]
    assert features["truth_distance_3d_m"].iloc[0] < 0.2
    assert features["truth_distance_3d_m"].iloc[1] > 10.0

    model = train_cluster_ranker(features, iterations=300, learning_rate=0.1)
    scores = predict_cluster_scores(features, model)

    assert scores[0] > scores[1]
    scored = score_cluster_candidates(candidates, model)
    assert "ranker_score" in scored.rows.columns
    assert scored.rows.sort_values("x_m")["ranker_score"].iloc[0] > scored.rows.sort_values("x_m")[
        "ranker_score"
    ].iloc[1]


def test_cross_sensor_candidate_merging_creates_extra_candidate() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqA", "seqA"],
                "time_s": [1.0, 1.02, 1.0],
                "source": ["lidar_360", "livox_avia", "radar_enhance_pcl"],
                "track_id": ["a", "b", "c"],
                "x_m": [5.0, 5.2, 20.0],
                "y_m": [1.0, 1.1, 20.0],
                "z_m": [2.0, 2.1, 4.0],
                "std_xy_m": [0.5, 0.5, 0.5],
                "std_z_m": [0.5, 0.5, 0.5],
                "confidence": [4.0, 3.0, 2.0],
                "class_name": ["uav", "uav", "uav"],
            }
        )
    )

    merged = merge_cross_sensor_candidate_clusters(
        candidates,
        time_window_s=0.05,
        distance_gate_m=1.0,
    )

    assert len(merged.rows) == 1
    row = merged.rows.iloc[0]
    assert row["source"] == "cross-sensor-merged"
    assert row["cross_sensor_neighbor_count"] == 2
    assert 5.0 < row["x_m"] < 5.2


def test_cluster_ranker_cli_trains_and_scores(tmp_path: Path) -> None:
    candidates = point_rows_to_candidates(_point_rows(), voxel_size_m=0.5, min_points=3)
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    model_json = tmp_path / "model.json"
    scored_csv = tmp_path / "scored.csv"
    train_features_csv = tmp_path / "train_features.csv"
    candidates.rows.to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    assert (
        cluster_ranker_main(
            [
                "--train-candidates",
                str(candidates_csv),
                "--train-truth",
                str(truth_csv),
                "--score-candidates",
                str(candidates_csv),
                "--model-json",
                str(model_json),
                "--train-features-csv",
                str(train_features_csv),
                "--scored-candidates-csv",
                str(scored_csv),
                "--good-threshold-m",
                "1.0",
                "--max-truth-time-delta-s",
                "0.1",
            ]
        )
        == 0
    )

    assert model_json.exists()
    assert scored_csv.exists()
    assert train_features_csv.exists()
    model = load_cluster_ranker_model(model_json)
    assert model.feature_columns
    scored = pd.read_csv(scored_csv).sort_values("x_m")
    assert scored["ranker_score"].iloc[0] > scored["ranker_score"].iloc[1]
    assert json.loads(model_json.read_text(encoding="utf-8"))["model_type"] == "logistic"


def test_cluster_ranker_model_round_trips(tmp_path: Path) -> None:
    features = build_cluster_feature_table(
        point_rows_to_candidates(_point_rows(), voxel_size_m=0.5, min_points=3),
        truth=_truth_rows(),
        good_threshold_m=1.0,
        max_truth_time_delta_s=0.1,
    )
    model = train_cluster_ranker(features, iterations=20)
    path = save_cluster_ranker_model(model, tmp_path / "model.json")

    loaded = load_cluster_ranker_model(path)

    assert loaded.feature_columns == model.feature_columns
    assert loaded.source_values == model.source_values


def test_mmuad_cli_applies_cluster_ranker_model(tmp_path: Path) -> None:
    candidates = point_rows_to_candidates(_point_rows(), voxel_size_m=0.5, min_points=3)
    features = build_cluster_feature_table(
        candidates,
        truth=_truth_rows(),
        good_threshold_m=1.0,
        max_truth_time_delta_s=0.1,
    )
    model = train_cluster_ranker(features, iterations=80, learning_rate=0.1)
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    model_json = save_cluster_ranker_model(model, tmp_path / "model.json")
    scored_csv = tmp_path / "scored.csv"
    features_csv = tmp_path / "features.csv"
    merged_csv = tmp_path / "merged.csv"
    output_dir = tmp_path / "out"
    candidates.rows.to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = mmuad_cli_main(
        [
            "--candidate-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--cluster-ranker-model-json",
            str(model_json),
            "--cluster-ranker-scored-candidates-csv",
            str(scored_csv),
            "--cluster-ranker-score-features-csv",
            str(features_csv),
            "--cluster-ranker-merged-candidates-csv",
            str(merged_csv),
        ]
    )

    assert status == 0
    scored = pd.read_csv(scored_csv).sort_values("x_m")
    feature_rows = pd.read_csv(features_csv).sort_values("x_m")
    assert "ranker_score" in scored.columns
    assert "ranker_score" in feature_rows.columns
    assert scored["confidence"].iloc[0] > scored["confidence"].iloc[1]
    assert (output_dir / "mmuad_estimates.csv").exists()
