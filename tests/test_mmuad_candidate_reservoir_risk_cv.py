from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_risk_cv import (
    build_risk_aggregate_summary,
    main as risk_cv_main,
    select_candidate_reservoir_offsets_by_risk_cv,
)


def _risk_candidate_rows() -> pd.DataFrame:
    records = []
    translated_error = {"seqA": 0.0, "seqB": 0.0, "seqC": 10.0}
    for sequence_id in ("seqA", "seqB", "seqC"):
        records.extend(
            [
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "lidar_360",
                    "track_id": f"{sequence_id}-raw",
                    "candidate_branch": "raw",
                    "x_m": 6.0,
                    "y_m": 0.0,
                    "z_m": 0.0,
                    "ranker_score": 0.1,
                    "confidence": 0.1,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "livox_avia",
                    "track_id": f"{sequence_id}-translated",
                    "candidate_branch": "translated",
                    "x_m": translated_error[sequence_id],
                    "y_m": 0.0,
                    "z_m": 0.0,
                    "ranker_score": 0.9,
                    "confidence": 0.9,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB", "seqC"],
            "time_s": [0.0, 0.0, 0.0],
            "x_m": [0.0, 0.0, 0.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
        }
    )


def _selector_kwargs() -> dict[str, object]:
    return {
        "branch_offset_grid": ["raw=0,1"],
        "global_top_n": 1,
        "per_source_top_n": 0,
        "per_branch_top_n": 0,
        "max_candidates_per_frame": 1,
        "top_k_values": (1,),
        "max_truth_time_delta_s": 0.1,
        "selection_metric": "oracle_top1_3d_m_mse",
        "tail_quantile": 1.0,
        "write_best_reservoir": True,
    }


def test_risk_aggregate_trades_mean_for_worst_sequence() -> None:
    folds = pd.DataFrame(
        {
            "grid_label": ["risky", "risky", "risky", "stable", "stable", "stable"],
            "branch_score_offsets_json": ["{}"] * 6,
            "source_score_offsets_json": ["{}"] * 6,
            "oracle_top1_3d_m_mse": [0.0, 0.0, 100.0, 36.0, 36.0, 36.0],
        }
    )
    mean_only = build_risk_aggregate_summary(
        folds,
        selection_metric="oracle_top1_3d_m_mse",
        risk_aversion=0.0,
    )
    risk_aware = build_risk_aggregate_summary(
        folds,
        selection_metric="oracle_top1_3d_m_mse",
        risk_aversion=0.5,
    )

    assert mean_only.iloc[0]["grid_label"] == "risky"
    assert risk_aware.iloc[0]["grid_label"] == "stable"
    assert risk_aware.iloc[0]["oracle_top1_3d_m_mse_risk_score"] == pytest.approx(36.0)


def test_risk_cv_selects_stable_raw_branch() -> None:
    mean_config, _, _, mean_best = select_candidate_reservoir_offsets_by_risk_cv(
        _risk_candidate_rows(),
        _truth_rows(),
        risk_aversion=0.0,
        **_selector_kwargs(),
    )
    risk_config, folds, aggregate, risk_best = select_candidate_reservoir_offsets_by_risk_cv(
        _risk_candidate_rows(),
        _truth_rows(),
        risk_aversion=0.5,
        **_selector_kwargs(),
    )

    assert mean_config["selected_grid_label"] == "identity"
    assert mean_best is not None
    assert set(mean_best["candidate_branch"]) == {"translated"}
    assert risk_config["selected_grid_label"] == "branch_raw_1"
    assert risk_config["branch_score_offsets"] == {"raw": 1.0}
    assert risk_config["selected_metric_value"] == pytest.approx(36.0)
    assert risk_config["selected_metric_max"] == pytest.approx(36.0)
    assert risk_config["selected_risk_score"] == pytest.approx(36.0)
    assert folds["holdout_sequence_id"].nunique() == 3
    assert aggregate.iloc[0]["grid_label"] == "branch_raw_1"
    assert risk_best is not None
    assert set(risk_best["candidate_branch"]) == {"raw"}


def test_risk_cv_cli_writes_frozen_config(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _risk_candidate_rows().to_csv(candidate_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = risk_cv_main(
        [
            "--candidate",
            f"mixed={candidate_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--branch-score-offset-grid",
            "raw=0,1",
            "--global-top-n",
            "1",
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "0",
            "--max-candidates-per-frame",
            "1",
            "--top-k",
            "1",
            "--selection-metric",
            "oracle_top1_3d_m_mse",
            "--max-truth-time-delta-s",
            "0.1",
            "--risk-aversion",
            "0.5",
            "--tail-quantile",
            "1",
            "--write-best-reservoir",
        ]
    )

    assert status == 0
    config_path = output_dir / "mmuad_candidate_reservoir_risk_cv_selected_config.json"
    aggregate_path = output_dir / "mmuad_candidate_reservoir_risk_cv_aggregate.csv"
    reservoir_path = output_dir / "mmuad_candidate_reservoir_risk_cv_selected.csv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["selected_grid_label"] == "branch_raw_1"
    assert config["risk_aversion"] == 0.5
    assert config["tail_quantile"] == 1.0
    assert aggregate_path.exists()
    assert reservoir_path.exists()


def test_risk_parameters_must_be_finite_and_bounded() -> None:
    folds = pd.DataFrame(
        {
            "grid_label": ["identity"],
            "oracle_top1_3d_m_mse": [1.0],
        }
    )
    with pytest.raises(ValueError, match="risk_aversion"):
        build_risk_aggregate_summary(
            folds,
            selection_metric="oracle_top1_3d_m_mse",
            risk_aversion=float("nan"),
        )
    with pytest.raises(ValueError, match="tail_quantile"):
        build_risk_aggregate_summary(
            folds,
            selection_metric="oracle_top1_3d_m_mse",
            tail_quantile=1.1,
        )
