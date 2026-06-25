from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_consensus_uncertainty import (
    apply_consensus_conditioned_uncertainty,
    attach_consensus_uncertainty_features,
    main as consensus_uncertainty_main,
    train_consensus_conditioned_uncertainty,
)
from raft_uav.mmuad.candidate_uncertainty import predict_candidate_sigma
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_s in range(6):
        true_x = float(time_s * 2)
        rows.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"lidar-good-{time_s}",
                    "candidate_branch": "raw",
                    "candidate_origin_row": f"lidar-{time_s}",
                    "x_m": true_x,
                    "y_m": 0.0,
                    "z_m": 2.0,
                    "confidence": 0.7,
                    "ranker_score": 0.7,
                    "cluster_point_count": 20,
                    "cluster_extent_3d_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s) + 0.01,
                    "source": "livox_avia",
                    "track_id": f"livox-good-{time_s}",
                    "candidate_branch": "dynamic",
                    "candidate_origin_row": f"livox-{time_s}",
                    "x_m": true_x + 0.2,
                    "y_m": 0.1,
                    "z_m": 2.0,
                    "confidence": 0.65,
                    "ranker_score": 0.65,
                    "cluster_point_count": 15,
                    "cluster_extent_3d_m": 1.2,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "radar_enhance_pcl",
                    "track_id": f"radar-bad-{time_s}",
                    "candidate_branch": "raw",
                    "candidate_origin_row": f"radar-{time_s}",
                    "x_m": true_x + 20.0,
                    "y_m": 10.0,
                    "z_m": 8.0,
                    "confidence": 0.8,
                    "ranker_score": 0.8,
                    "cluster_point_count": 3,
                    "cluster_extent_3d_m": 8.0,
                },
            ]
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


def test_consensus_features_are_exposed_to_uncertainty_model() -> None:
    model, augmented, features, summary = train_consensus_conditioned_uncertainty(
        CandidateFrame(_candidate_rows()),
        _truth_rows(),
        model_type="ridge",
        sigma_min_m=1.0,
        sigma_max_m=30.0,
        ridge_alpha=0.1,
        max_truth_time_delta_s=0.05,
        time_window_s=0.05,
        distance_gate_m=3.0,
    )

    assert "candidate_reservoir_consensus_score" in model.feature_columns
    assert "candidate_reservoir_consensus_nearest_cross_source_distance_m" in model.feature_columns
    assert summary["consensus_feature_count"] > 0
    assert summary["consensus_features_used_by_model"]
    assert "branch_consensus_score" in augmented.rows.columns
    assert "candidate_reservoir_consensus_score" in features.columns

    predicted = pd.Series(predict_candidate_sigma(features, model), index=features.index)
    good = predicted.loc[~features["track_id"].astype(str).str.contains("bad")]
    bad = predicted.loc[features["track_id"].astype(str).str.contains("bad")]
    assert float(bad.mean()) > float(good.mean())
    assert np.all((predicted >= 1.0) & (predicted <= 30.0))


def test_consensus_conditioned_uncertainty_apply_is_truth_free() -> None:
    model, _, _, _ = train_consensus_conditioned_uncertainty(
        CandidateFrame(_candidate_rows()),
        _truth_rows(),
        model_type="ridge",
        max_truth_time_delta_s=0.05,
    )

    applied = apply_consensus_conditioned_uncertainty(
        CandidateFrame(_candidate_rows()),
        model,
        replace_covariance=True,
        z_scale=2.0,
    ).rows

    assert applied["predicted_sigma_m"].notna().all()
    assert (applied["std_xy_m"] == applied["predicted_sigma_m"]).all()
    assert (applied["std_z_m"] == 2.0 * applied["predicted_sigma_m"]).all()
    assert "candidate_reservoir_consensus_score" in applied.columns


def test_consensus_uncertainty_cli_trains_and_applies(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    model_json = tmp_path / "model.json"
    features_csv = tmp_path / "features.csv"
    augmented_csv = tmp_path / "augmented.csv"
    summary_json = tmp_path / "summary.json"
    output_csv = tmp_path / "scored.csv"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    train_status = consensus_uncertainty_main(
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
            "--augmented-candidates-csv",
            str(augmented_csv),
            "--summary-json",
            str(summary_json),
            "--model-type",
            "ridge",
            "--max-truth-time-delta-s",
            "0.05",
        ]
    )
    apply_status = consensus_uncertainty_main(
        [
            "apply",
            "--candidates-csv",
            str(candidates_csv),
            "--model-json",
            str(model_json),
            "--output-csv",
            str(output_csv),
        ]
    )

    assert train_status == 0
    assert apply_status == 0
    assert model_json.exists()
    assert features_csv.exists()
    assert augmented_csv.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["consensus_feature_count"] > 0
    assert summary["consensus_features_used_by_model"]
    output = pd.read_csv(output_csv)
    assert output["predicted_sigma_m"].notna().all()
    assert "candidate_reservoir_consensus_score" in output.columns


def test_consensus_feature_aliases_skip_non_numeric_columns() -> None:
    augmented, aliases = attach_consensus_uncertainty_features(
        CandidateFrame(_candidate_rows()),
    )

    assert aliases
    assert all(column.startswith("candidate_reservoir_consensus_") for column in aliases)
    assert "candidate_reservoir_consensus_nearest_cross_source" not in augmented.rows.columns
