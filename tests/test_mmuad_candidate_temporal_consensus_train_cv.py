from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus import TemporalConsensusConfig
from raft_uav.mmuad.candidate_temporal_consensus_train_cv import (
    apply_main,
    load_temporal_consensus_selection,
    select_main,
    select_temporal_consensus_config_by_sequence_cv,
)
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sequence_id in ("seqA", "seqB", "seqC"):
        records.extend(
            [
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "livox_avia",
                    "candidate_branch": "dynamic",
                    "track_id": f"{sequence_id}-previous",
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.5,
                    "confidence": 0.5,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 1.0,
                    "source": "lidar_360",
                    "candidate_branch": "raw",
                    "track_id": f"{sequence_id}-smooth",
                    "x_m": 1.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.1,
                    "confidence": 0.1,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 1.0,
                    "source": "radar_enhance_pcl",
                    "candidate_branch": "translated",
                    "track_id": f"{sequence_id}-outlier",
                    "x_m": 8.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.99,
                    "confidence": 0.99,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 2.0,
                    "source": "livox_avia",
                    "candidate_branch": "dynamic",
                    "track_id": f"{sequence_id}-next",
                    "x_m": 2.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.5,
                    "confidence": 0.5,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sequence_id in ("seqA", "seqB", "seqC"):
        for time_s in (0.0, 1.0, 2.0):
            records.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": time_s,
                    "x_m": time_s,
                    "y_m": 0.0,
                    "z_m": 1.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _configs() -> tuple[TemporalConsensusConfig, TemporalConsensusConfig]:
    good = TemporalConsensusConfig(
        max_time_gap_s=1.1,
        max_speed_mps=4.0,
        distance_scale_m=2.0,
        acceleration_scale_mps2=5.0,
        base_score_weight=0.0,
    )
    bad = TemporalConsensusConfig(
        max_time_gap_s=1.1,
        max_speed_mps=4.0,
        distance_scale_m=2.0,
        acceleration_scale_mps2=5.0,
        base_score_weight=10.0,
    )
    return good, bad


def test_train_cv_selects_temporally_consistent_config() -> None:
    selected, folds, grid, candidates, provenance = (
        select_temporal_consensus_config_by_sequence_cv(
            CandidateFrame(_candidate_rows()),
            _truth_rows(),
            configs=_configs(),
            selection_metric="top1_3d_m_mse",
            max_truth_time_delta_s=0.01,
        )
    )

    assert selected.base_score_weight == pytest.approx(0.0)
    assert len(folds) == 3
    assert folds["selected_config_index"].eq(1).all()
    assert grid.loc[0, "selection_rank"] == 1
    assert grid.loc[0, "top1_3d_m_mse"] == pytest.approx(0.0)
    assert grid.loc[1, "top1_3d_m_mse"] > 0.0
    assert provenance["selected_config_index"] == 1
    assert "candidate_temporal_consensus_score" in candidates.rows.columns


def test_selection_json_round_trip_accepts_wrapped_payload(tmp_path: Path) -> None:
    selected, _, _, _, provenance = select_temporal_consensus_config_by_sequence_cv(
        CandidateFrame(_candidate_rows()),
        _truth_rows(),
        configs=_configs(),
        max_truth_time_delta_s=0.01,
    )
    config_json = tmp_path / "selection.json"
    config_json.write_text(json.dumps(provenance), encoding="utf-8")

    loaded = load_temporal_consensus_selection(config_json)

    assert loaded == selected


def test_train_cv_and_apply_clis_write_reusable_artifacts(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "train_cv"
    applied_csv = tmp_path / "applied.csv"
    applied_summary = tmp_path / "applied_summary.json"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = select_main(
        [
            "--candidate-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--max-truth-time-delta-s",
            "0.01",
            "--max-time-gap-s",
            "1.1",
            "--max-speed-mps",
            "4",
            "--distance-scale-m",
            "2",
            "--base-score-weight",
            "0",
            "--base-score-weight",
            "10",
            "--bidirectional-bonus",
            "0.75",
            "--write-selected-candidates",
        ]
    )

    config_json = output_dir / "mmuad_temporal_consensus_train_selected_config.json"
    folds_csv = output_dir / "mmuad_temporal_consensus_train_cv_folds.csv"
    grid_csv = output_dir / "mmuad_temporal_consensus_train_grid_summary.csv"
    selected_csv = output_dir / "mmuad_temporal_consensus_train_selected_candidates.csv"
    assert status == 0
    assert config_json.exists()
    assert folds_csv.exists()
    assert grid_csv.exists()
    assert selected_csv.exists()

    apply_status = apply_main(
        [
            "--candidate-csv",
            str(candidates_csv),
            "--config-json",
            str(config_json),
            "--output-csv",
            str(applied_csv),
            "--summary-json",
            str(applied_summary),
            "--replace-confidence",
        ]
    )

    assert apply_status == 0
    applied = pd.read_csv(applied_csv)
    summary = json.loads(applied_summary.read_text(encoding="utf-8"))
    selected_config = load_temporal_consensus_selection(config_json)
    assert selected_config.base_score_weight == pytest.approx(0.0)
    assert "candidate_temporal_consensus_score" in applied.columns
    assert "raw_confidence" in applied.columns
    assert summary["config"]["base_score_weight"] == pytest.approx(0.0)
