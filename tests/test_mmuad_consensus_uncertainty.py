from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from raft_uav.mmuad.consensus_uncertainty import (
    apply_consensus_conditioned_uncertainty,
    attach_consensus_uncertainty_features,
    main as consensus_uncertainty_main,
    train_consensus_conditioned_uncertainty,
)
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_s in range(4):
        truth_x = float(time_s * 2)
        origin = int(time_s)
        rows.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"raw-{time_s}",
                    "candidate_branch": "raw",
                    "mmuad_calibration_origin_row": origin,
                    "x_m": truth_x + 8.0,
                    "y_m": 0.0,
                    "z_m": 2.0,
                    "std_xy_m": 10.0,
                    "std_z_m": 10.0,
                    "ranker_score": 0.9,
                    "confidence": 0.9,
                    "cluster_point_count": 20,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"calibrated-{time_s}",
                    "candidate_branch": "source_translation",
                    "mmuad_calibration_origin_row": origin,
                    "x_m": truth_x,
                    "y_m": 0.0,
                    "z_m": 2.0,
                    "std_xy_m": 10.0,
                    "std_z_m": 10.0,
                    "ranker_score": 0.4,
                    "confidence": 0.4,
                    "cluster_point_count": 20,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "livox_avia",
                    "track_id": f"livox-{time_s}",
                    "candidate_branch": "raw",
                    "mmuad_calibration_origin_row": 100 + origin,
                    "x_m": truth_x + 0.2,
                    "y_m": 0.0,
                    "z_m": 2.0,
                    "std_xy_m": 10.0,
                    "std_z_m": 10.0,
                    "ranker_score": 0.5,
                    "confidence": 0.5,
                    "cluster_point_count": 15,
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [float(value) for value in range(4)],
            "x_m": [float(value * 2) for value in range(4)],
            "y_m": [0.0] * 4,
            "z_m": [2.0] * 4,
        }
    )


def test_consensus_features_are_consumed_by_uncertainty_model() -> None:
    model, features, augmented = train_consensus_conditioned_uncertainty(
        CandidateFrame(_candidate_rows()),
        _truth_rows(),
        model_type="ridge",
        sigma_min_m=1.0,
        sigma_max_m=30.0,
        ridge_alpha=0.1,
        max_truth_time_delta_s=0.01,
        distance_gate_m=2.0,
        distance_scale_m=2.0,
    )

    assert "image_branch_consensus_score" in model.feature_columns
    assert "image_branch_consensus_neighbor_count" in model.feature_columns
    assert "image_branch_consensus_score" in features.columns
    assert "branch_consensus_score" in augmented.rows.columns


def test_consensus_shrinkage_rewards_independent_sensor_support() -> None:
    model, _, _ = train_consensus_conditioned_uncertainty(
        CandidateFrame(_candidate_rows()),
        _truth_rows(),
        model_type="ridge",
        sigma_min_m=1.0,
        sigma_max_m=30.0,
        ridge_alpha=0.1,
        max_truth_time_delta_s=0.01,
        distance_gate_m=2.0,
        distance_scale_m=2.0,
    )
    applied = apply_consensus_conditioned_uncertainty(
        CandidateFrame(_candidate_rows()),
        model,
        consensus_sigma_weight=1.0,
        consensus_sigma_min_factor=0.5,
        distance_gate_m=2.0,
        distance_scale_m=2.0,
        replace_covariance=True,
        z_scale=2.0,
    ).rows.set_index("track_id")

    raw = applied.loc["raw-0"]
    calibrated = applied.loc["calibrated-0"]
    assert calibrated["candidate_uncertainty_consensus_factor"] < raw[
        "candidate_uncertainty_consensus_factor"
    ]
    assert calibrated["predicted_sigma_m"] <= calibrated["raw_predicted_sigma_m"]
    assert raw["predicted_sigma_m"] <= raw["raw_predicted_sigma_m"]
    assert np.allclose(applied["std_xy_m"], applied["predicted_sigma_m"])
    assert np.allclose(applied["std_z_m"], 2.0 * applied["predicted_sigma_m"])


def test_consensus_uncertainty_cli_trains_and_applies(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    model_json = tmp_path / "model.json"
    features_csv = tmp_path / "features.csv"
    train_summary_json = tmp_path / "train_summary.json"
    output_csv = tmp_path / "uncertain_candidates.csv"
    provenance_json = tmp_path / "apply_summary.json"
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
            "--summary-json",
            str(train_summary_json),
            "--model-type",
            "ridge",
            "--max-truth-time-delta-s",
            "0.01",
            "--distance-gate-m",
            "2",
            "--distance-scale-m",
            "2",
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
            "--provenance-json",
            str(provenance_json),
            "--consensus-sigma-weight",
            "1",
            "--distance-gate-m",
            "2",
            "--distance-scale-m",
            "2",
        ]
    )

    assert train_status == 0
    assert apply_status == 0
    train_summary = json.loads(train_summary_json.read_text(encoding="utf-8"))
    assert "image_branch_consensus_score" in train_summary["consensus_feature_columns"]
    output = pd.read_csv(output_csv)
    assert output["predicted_sigma_m"].notna().all()
    assert "candidate_uncertainty_consensus_factor" in output.columns
    apply_summary = json.loads(provenance_json.read_text(encoding="utf-8"))
    assert apply_summary["row_count"] == len(output)


def test_consensus_uncertainty_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-consensus-uncertainty"]
        == "raft_uav.mmuad.consensus_uncertainty:main"
    )
