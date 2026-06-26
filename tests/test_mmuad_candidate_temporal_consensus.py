from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus import (
    TemporalConsensusConfig,
    add_temporal_candidate_consensus,
    main as temporal_consensus_main,
)
from raft_uav.mmuad.candidate_uncertainty import train_candidate_uncertainty
from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 1.0, 2.0],
            "source": ["livox_avia", "lidar_360", "radar_enhance_pcl", "livox_avia"],
            "candidate_branch": ["dynamic", "raw", "source_translation", "dynamic"],
            "track_id": ["previous", "smooth", "outlier", "next"],
            "x_m": [0.0, 1.0, 50.0, 2.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "confidence": [0.5, 0.1, 0.99, 0.5],
            "ranker_score": [0.5, 0.1, 0.99, 0.5],
            "cluster_point_count": [10, 10, 50, 10],
        }
    )


def _config() -> TemporalConsensusConfig:
    return TemporalConsensusConfig(
        max_time_gap_s=1.1,
        max_speed_mps=10.0,
        distance_scale_m=2.0,
        acceleration_scale_mps2=5.0,
    )


def test_temporal_consensus_promotes_smooth_bidirectional_candidate() -> None:
    augmented = add_temporal_candidate_consensus(
        CandidateFrame(_candidate_rows()),
        config=_config(),
    ).rows
    middle = augmented.loc[augmented["time_s"] == 1.0].set_index("track_id")

    assert middle.loc["smooth", "candidate_reservoir_temporal_bidirectional"] == pytest.approx(
        1.0
    )
    assert middle.loc["smooth", "candidate_reservoir_temporal_interpolation_residual_m"] == (
        pytest.approx(0.0)
    )
    assert middle.loc["smooth", "candidate_reservoir_temporal_acceleration_mps2"] == (
        pytest.approx(0.0)
    )
    assert middle.loc["smooth", "candidate_temporal_consensus_score"] > middle.loc[
        "outlier", "candidate_temporal_consensus_score"
    ]
    assert pd.isna(
        middle.loc["outlier", "candidate_reservoir_temporal_backward_distance_m"]
    )


def test_temporal_consensus_records_cross_source_and_branch_support() -> None:
    augmented = add_temporal_candidate_consensus(
        CandidateFrame(_candidate_rows()),
        config=_config(),
    ).rows
    smooth = augmented.loc[augmented["track_id"] == "smooth"].iloc[0]

    assert smooth["candidate_reservoir_temporal_other_source_support_count"] == pytest.approx(
        2.0
    )
    assert smooth["candidate_reservoir_temporal_other_branch_support_count"] == pytest.approx(
        2.0
    )
    assert smooth["candidate_temporal_backward_source"] == "livox_avia"
    assert smooth["candidate_temporal_forward_branch"] == "dynamic"


def test_temporal_features_feed_candidate_uncertainty() -> None:
    augmented = add_temporal_candidate_consensus(
        CandidateFrame(_candidate_rows()),
        config=_config(),
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )
    features = build_cluster_feature_table(
        augmented,
        truth=truth,
        max_truth_time_delta_s=0.1,
    )
    model = train_candidate_uncertainty(
        features,
        model_type="ridge",
        target_transform="log1p",
    )

    assert "candidate_reservoir_temporal_score" in model.feature_columns
    assert "candidate_reservoir_temporal_interpolation_residual_m" in model.feature_columns


def test_temporal_consensus_cli_writes_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "temporal.csv"
    summary_json = tmp_path / "summary.json"
    _candidate_rows().to_csv(candidates_csv, index=False)

    status = temporal_consensus_main(
        [
            "--candidate-csv",
            str(candidates_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--max-time-gap-s",
            "1.1",
            "--max-speed-mps",
            "10",
            "--replace-confidence",
        ]
    )

    assert status == 0
    rows = pd.read_csv(output_csv)
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "candidate_temporal_consensus_score" in rows.columns
    assert "raw_confidence" in rows.columns
    assert payload["summary"]["bidirectional_supported_count"] == 1
    assert payload["config"]["max_speed_mps"] == pytest.approx(10.0)
