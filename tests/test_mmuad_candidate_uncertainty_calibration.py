from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_uncertainty_calibration import (
    apply_candidate_sigma_calibration,
    apply_main,
    fit_candidate_sigma_calibration,
    fit_main,
    load_candidate_sigma_calibration,
    save_candidate_sigma_calibration,
)
from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    branch_offsets = {
        ("lidar_360", "raw"): 5.0,
        ("lidar_360", "translated"): 20.0,
        ("radar_enhance_pcl", "raw"): 10.0,
        ("radar_enhance_pcl", "translated"): 30.0,
    }
    for time_s in range(4):
        for (source, branch), offset_m in branch_offsets.items():
            records.append(
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": source,
                    "track_id": f"{source}-{branch}-{time_s}",
                    "candidate_branch": branch,
                    "x_m": float(time_s) + offset_m,
                    "y_m": 0.0,
                    "z_m": 2.0,
                    "confidence": 0.5,
                    "predicted_sigma_m": 10.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [float(value) for value in range(4)],
            "x_m": [float(value) for value in range(4)],
            "y_m": [0.0] * 4,
            "z_m": [2.0] * 4,
        }
    )


def _training_features() -> pd.DataFrame:
    return build_cluster_feature_table(
        CandidateFrame(_candidate_rows()),
        truth=_truth_rows(),
        max_truth_time_delta_s=0.01,
    )


def test_fit_candidate_sigma_calibration_learns_source_branch_scales() -> None:
    calibration = fit_candidate_sigma_calibration(
        _training_features(),
        target_quantile=0.5,
        min_group_rows=1,
        shrinkage_rows=0.0,
        scale_min=0.1,
        scale_max=5.0,
    )

    raw_lidar_key = json.dumps(["lidar_360", "raw"], separators=(",", ":"))
    translated_lidar_key = json.dumps(
        ["lidar_360", "translated"],
        separators=(",", ":"),
    )
    raw_radar_key = json.dumps(
        ["radar_enhance_pcl", "raw"],
        separators=(",", ":"),
    )
    translated_radar_key = json.dumps(
        ["radar_enhance_pcl", "translated"],
        separators=(",", ":"),
    )

    assert calibration.source_branch_scales[raw_lidar_key] == 0.5
    assert calibration.source_branch_scales[translated_lidar_key] == 2.0
    assert calibration.source_branch_scales[raw_radar_key] == 1.0
    assert calibration.source_branch_scales[translated_radar_key] == 3.0
    assert calibration.global_scale == 1.5


def test_apply_candidate_sigma_calibration_uses_hierarchical_backoff() -> None:
    calibration = fit_candidate_sigma_calibration(
        _training_features(),
        target_quantile=0.5,
        min_group_rows=1,
        shrinkage_rows=0.0,
        scale_min=0.1,
        scale_max=5.0,
    )
    inference = pd.DataFrame(
        {
            "sequence_id": ["seqB", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 2.0],
            "source": ["lidar_360", "lidar_360", "unknown_sensor"],
            "track_id": ["known", "source-backoff", "global-backoff"],
            "candidate_branch": ["raw", "new_branch", "new_branch"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [2.0, 2.0, 2.0],
            "predicted_sigma_m": [10.0, 10.0, 10.0],
            "std_xy_m": [10.0, 10.0, 10.0],
            "std_z_m": [10.0, 10.0, 10.0],
        }
    )

    calibrated = apply_candidate_sigma_calibration(
        CandidateFrame(inference),
        calibration,
        replace_covariance=True,
        z_scale=2.0,
    ).rows.sort_values("time_s")

    assert calibrated.iloc[0]["candidate_sigma_calibration_level"] == "source_branch"
    assert calibrated.iloc[1]["candidate_sigma_calibration_level"] == "source"
    assert calibrated.iloc[2]["candidate_sigma_calibration_level"] == "global"
    np.testing.assert_allclose(
        calibrated["std_xy_m"],
        calibrated["calibrated_sigma_m"],
    )
    np.testing.assert_allclose(
        calibrated["std_z_m"],
        calibrated["calibrated_sigma_m"] * 2.0,
    )
    assert (calibrated["candidate_sigma_uncalibrated_m"] == 10.0).all()


def test_candidate_sigma_calibration_round_trip_and_cli(tmp_path: Path) -> None:
    features = _training_features()
    calibration = fit_candidate_sigma_calibration(
        features,
        min_group_rows=1,
        shrinkage_rows=0.0,
    )
    direct_json = tmp_path / "direct_calibration.json"
    save_candidate_sigma_calibration(calibration, direct_json)
    loaded = load_candidate_sigma_calibration(direct_json)
    assert loaded == calibration

    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    calibration_json = tmp_path / "calibration.json"
    features_csv = tmp_path / "features.csv"
    summary_json = tmp_path / "summary.json"
    output_csv = tmp_path / "calibrated_candidates.csv"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    assert (
        fit_main(
            [
                "--candidates-csv",
                str(candidates_csv),
                "--truth-csv",
                str(truth_csv),
                "--calibration-json",
                str(calibration_json),
                "--features-csv",
                str(features_csv),
                "--summary-json",
                str(summary_json),
                "--min-group-rows",
                "1",
                "--shrinkage-rows",
                "0",
                "--max-truth-time-delta-s",
                "0.01",
            ]
        )
        == 0
    )
    assert (
        apply_main(
            [
                "--candidates-csv",
                str(candidates_csv),
                "--calibration-json",
                str(calibration_json),
                "--output-csv",
                str(output_csv),
                "--replace-covariance",
            ]
        )
        == 0
    )

    assert calibration_json.exists()
    assert features_csv.exists()
    assert summary_json.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["row_count"] == len(_candidate_rows())
    output = pd.read_csv(output_csv)
    assert output["calibrated_sigma_m"].notna().all()
    assert output["candidate_sigma_calibration_level"].eq("source_branch").all()
