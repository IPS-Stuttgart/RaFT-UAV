from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus_train_cv import (
    apply_main,
    main,
    select_temporal_consensus_config_by_sequence_cv,
)


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[dict[str, object]] = []
    truth: list[dict[str, object]] = []
    for sequence_id, offset in (("a", 0.0), ("b", 10.0), ("c", 20.0)):
        for time_s in (0.0, 1.0, 2.0):
            truth.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": time_s,
                    "x_m": offset + time_s,
                    "y_m": 0.0,
                    "z_m": 1.0,
                }
            )
        candidates.extend(
            [
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "livox",
                    "candidate_branch": "dynamic",
                    "track_id": f"{sequence_id}-prev",
                    "x_m": offset,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.5,
                    "confidence": 0.5,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 1.0,
                    "source": "lidar",
                    "candidate_branch": "raw",
                    "track_id": f"{sequence_id}-smooth",
                    "x_m": offset + 1.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.1,
                    "confidence": 0.1,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 1.0,
                    "source": "radar",
                    "candidate_branch": "translated",
                    "track_id": f"{sequence_id}-bad",
                    "x_m": offset + 50.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.99,
                    "confidence": 0.99,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 2.0,
                    "source": "livox",
                    "candidate_branch": "dynamic",
                    "track_id": f"{sequence_id}-next",
                    "x_m": offset + 2.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.5,
                    "confidence": 0.5,
                },
            ]
        )
    return pd.DataFrame(candidates), pd.DataFrame(truth)


def test_train_cv_recovers_smooth_candidate() -> None:
    candidates, truth = _rows()
    selected, folds, grid, augmented = select_temporal_consensus_config_by_sequence_cv(
        candidates,
        truth,
        base_score_weights=(0.25, 2.0),
        support_weights=(0.0, 1.0),
        bidirectional_bonuses=(0.0, 1.0),
        interpolation_weights=(0.0, 1.0),
        acceleration_weights=(0.0,),
        max_time_gap_s=1.1,
        max_speed_mps=10.0,
        distance_scale_m=2.0,
        acceleration_scale_mps2=5.0,
        source_diversity_bonus=0.0,
        branch_diversity_bonus=0.0,
        top_k_values=(1,),
        max_truth_time_delta_s=0.1,
        selection_metric="oracle_top1_3d_m_mse",
    )

    assert selected["selected_metric_value"] == pytest.approx(0.0)
    assert len(folds) == 3
    assert folds["holdout_selection_metric_value"].max() == pytest.approx(0.0)
    assert grid.iloc[0]["oracle_top1_3d_m_mse"] == pytest.approx(0.0)
    middle = augmented.loc[augmented["time_s"] == 1.0].set_index("track_id")
    assert middle.loc["a-smooth", "candidate_temporal_consensus_score"] > middle.loc[
        "a-bad", "candidate_temporal_consensus_score"
    ]


def test_fit_and_apply_clis_write_truth_free_outputs(tmp_path: Path) -> None:
    candidates, truth = _rows()
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    fit_dir = tmp_path / "fit"
    candidates.to_csv(candidate_csv, index=False)
    truth.to_csv(truth_csv, index=False)

    assert (
        main(
            [
                "--candidate-csv",
                str(candidate_csv),
                "--truth-csv",
                str(truth_csv),
                "--output-dir",
                str(fit_dir),
                "--base-score-weight-grid",
                "0.25",
                "--support-weight-grid",
                "1",
                "--bidirectional-bonus-grid",
                "1",
                "--interpolation-weight-grid",
                "1",
                "--acceleration-weight-grid",
                "0",
                "--max-time-gap-s",
                "1.1",
                "--max-speed-mps",
                "10",
                "--top-k",
                "1",
                "--selection-metric",
                "oracle_top1_3d_m_mse",
            ]
        )
        == 0
    )
    config_json = fit_dir / "mmuad_temporal_consensus_train_selected_config.json"
    selected_config = json.loads(config_json.read_text(encoding="utf-8"))
    grid = pd.read_csv(fit_dir / "mmuad_temporal_consensus_train_grid_summary.csv")
    assert selected_config["top_k_values"] == [1]
    assert "oracle_top1_3d_m_mse" in grid.columns
    assert "oracle_top3_3d_m_mse" not in grid.columns

    output_csv = tmp_path / "applied.csv"
    summary_json = tmp_path / "summary.json"
    assert (
        apply_main(
            [
                "--config-json",
                str(config_json),
                "--candidate-csv",
                str(candidate_csv),
                "--output-csv",
                str(output_csv),
                "--summary-json",
                str(summary_json),
                "--replace-confidence",
            ]
        )
        == 0
    )
    applied = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "candidate_temporal_consensus_score" in applied.columns
    assert "raw_confidence" in applied.columns
    assert summary["truth_free"] is True
