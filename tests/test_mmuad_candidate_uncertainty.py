from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_uncertainty import (
    apply_candidate_uncertainty,
    load_candidate_uncertainty_model,
    main as uncertainty_main,
    predict_candidate_sigma,
    save_candidate_uncertainty_model,
    train_candidate_uncertainty,
)
from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_s in range(6):
        true_x = float(time_s * 2)
        rows.append(
            {
                "sequence_id": "seqA",
                "time_s": float(time_s),
                "source": "lidar_360",
                "track_id": f"good-{time_s}",
                "x_m": true_x,
                "y_m": 0.0,
                "z_m": 2.0,
                "std_xy_m": 10.0,
                "std_z_m": 10.0,
                "confidence": 0.95,
                "cluster_point_count": 30,
                "cluster_extent_3d_m": 1.0,
                "image_class_prob_2": 0.8,
                "image_class_prob_2_x_cluster_point_count": 24.0,
            }
        )
        rows.append(
            {
                "sequence_id": "seqA",
                "time_s": float(time_s),
                "source": "radar_enhance_pcl",
                "track_id": f"bad-{time_s}",
                "x_m": true_x + 20.0 + float(time_s),
                "y_m": 8.0,
                "z_m": 8.0,
                "std_xy_m": 10.0,
                "std_z_m": 10.0,
                "confidence": 0.05,
                "cluster_point_count": 2,
                "cluster_extent_3d_m": 8.0,
                "image_class_prob_2": 0.8,
                "image_class_prob_2_x_cluster_point_count": 1.6,
            }
        )
    return pd.DataFrame.from_records(rows)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6,
            "time_s": [float(value) for value in range(6)],
            "x_m": [float(value * 2) for value in range(6)],
            "y_m": [0.0] * 6,
            "z_m": [2.0] * 6,
        }
    )


def _training_features() -> pd.DataFrame:
    return build_cluster_feature_table(
        CandidateFrame(_candidate_rows()),
        truth=_truth_rows(),
        max_truth_time_delta_s=0.01,
    )


def test_ridge_uncertainty_predicts_larger_scale_for_bad_candidates() -> None:
    features = _training_features()
    model = train_candidate_uncertainty(
        features,
        model_type="ridge",
        target_transform="log1p",
        sigma_min_m=1.0,
        sigma_max_m=30.0,
        ridge_alpha=0.1,
    )
    predicted = pd.Series(predict_candidate_sigma(features, model), index=features.index)
    good = predicted.loc[features["track_id"].astype(str).str.startswith("good-")]
    bad = predicted.loc[features["track_id"].astype(str).str.startswith("bad-")]

    assert float(bad.mean()) > float(good.mean()) + 5.0
    assert "image_class_prob_2_x_cluster_point_count" in model.feature_columns
    assert np.all((predicted >= 1.0) & (predicted <= 30.0))


def test_uncertainty_model_round_trip_and_covariance_application(tmp_path: Path) -> None:
    features = _training_features()
    model = train_candidate_uncertainty(
        features,
        model_type="ridge",
        sigma_min_m=1.0,
        sigma_max_m=30.0,
    )
    model_path = tmp_path / "uncertainty_model.json"
    save_candidate_uncertainty_model(model, model_path)
    loaded = load_candidate_uncertainty_model(model_path)

    expected = predict_candidate_sigma(features, model)
    actual = predict_candidate_sigma(features, loaded)
    np.testing.assert_allclose(actual, expected)

    applied = apply_candidate_uncertainty(
        CandidateFrame(_candidate_rows()),
        loaded,
        replace_covariance=True,
        z_scale=2.0,
    ).rows
    assert applied["predicted_sigma_m"].notna().all()
    assert (applied["std_xy_m"] == applied["predicted_sigma_m"]).all()
    assert (applied["std_z_m"] == 2.0 * applied["predicted_sigma_m"]).all()
    assert (applied["raw_std_xy_m"] == 10.0).all()


def test_candidate_uncertainty_cli_trains_and_applies(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    model_json = tmp_path / "model.json"
    features_csv = tmp_path / "features.csv"
    summary_json = tmp_path / "summary.json"
    output_csv = tmp_path / "uncertain_candidates.csv"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    train_status = uncertainty_main(
        [
            "train",
            "--candidates-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--model-json",
            str(model_json),
            "--features-csv",
            str(features_csv),
            "--summary-json",
            str(summary_json),
            "--model-type",
            "ridge",
            "--max-truth-time-delta-s",
            "0.01",
        ]
    )
    apply_status = uncertainty_main(
        [
            "apply",
            "--candidates-csv",
            str(candidates_csv),
            "--model-json",
            str(model_json),
            "--output-csv",
            str(output_csv),
            "--replace-covariance",
        ]
    )

    assert train_status == 0
    assert apply_status == 0
    assert model_json.exists()
    assert features_csv.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["row_count"] == 12
    assert summary["model_type"] == "ridge"
    output = pd.read_csv(output_csv)
    assert output["predicted_sigma_m"].notna().all()
    assert (output["std_xy_m"] == output["predicted_sigma_m"]).all()
