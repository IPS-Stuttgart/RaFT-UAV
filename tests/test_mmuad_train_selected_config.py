from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.train_selected_config import build_train_selected_config
from raft_uav.mmuad.train_selected_config import load_train_selected_config
from raft_uav.mmuad.train_selected_config import main
from raft_uav.mmuad.train_selected_config import validate_train_selected_config
from raft_uav.mmuad.train_selected_config import write_train_selected_config


def test_train_selected_config_writes_required_fields_from_train_summaries(
    tmp_path: Path,
) -> None:
    source_csv = tmp_path / "source.csv"
    ranker_csv = tmp_path / "ranker.csv"
    reservoir_csv = tmp_path / "reservoir.csv"
    mixture_csv = tmp_path / "mixture.csv"
    viterbi_csv = tmp_path / "viterbi.csv"
    smoothing_csv = tmp_path / "smoothing.csv"
    classifier_csv = tmp_path / "classifier.csv"
    pd.DataFrame(
        [
            {"mode": "identity", "alpha": 0.0, "train_cv_pose_mse_loss_m2": 20.0},
            {"mode": "source-translation", "alpha": 0.5, "train_cv_pose_mse_loss_m2": 10.0},
        ]
    ).to_csv(source_csv, index=False)
    pd.DataFrame(
        [
            {
                "model_type": "sklearn-logistic",
                "target_column": "good_cluster_10m",
                "loso_pose_mse_loss_m2": 30.0,
            },
            {
                "model_type": "random-forest-classifier",
                "target_column": "good_cluster_5m",
                "point_extraction_mode": "static_dynamic_union",
                "loso_pose_mse_loss_m2": 15.0,
            },
        ]
    ).to_csv(ranker_csv, index=False)
    pd.DataFrame(
        [
            {
                "pose_mse_loss_m2": 50.0,
                "global_top_n": 40,
                "per_source_top_n": 3,
                "per_branch_top_n": 4,
                "max_candidates_per_frame": 60,
                "score_floor_quantile": 0.25,
                "cap_reason_bonus": 0.5,
            }
        ]
    ).to_csv(reservoir_csv, index=False)
    pd.DataFrame(
        [
            {
                "pose_mse_loss_m2": 45.0,
                "temperature": 64,
                "sigma_log_weight": 2.0,
                "huber_delta": 1.5,
                "smoothness_weight": 3600,
                "branch_balance": 0.25,
                "source_balance": 0.5,
            }
        ]
    ).to_csv(mixture_csv, index=False)
    pd.DataFrame(
        [
            {
                "motion_weight": 2.0,
                "ranker_weight": 1.0,
                "source_switch_penalty": 0.25,
                "max_speed_mps": 40.0,
                "gap_penalty": 0.1,
                "pose_mse_loss_m2": 9.0,
            }
        ]
    ).to_csv(viterbi_csv, index=False)
    pd.DataFrame(
        [
            {"mode": "none", "speed_gate_mps": 0.0, "blend": 1.0, "pose_mse_loss_m2": 12.0},
            {
                "mode": "fixed-lag",
                "speed_gate_mps": 20.0,
                "blend": 0.5,
                "pose_mse_loss_m2": 8.0,
            },
        ]
    ).to_csv(smoothing_csv, index=False)
    pd.DataFrame(
        [
            {"method": "random-forest", "fusion_weight": 0.5, "classification_accuracy": 0.75},
            {
                "method": "hist-gradient-boosting",
                "fusion_weight": 0.25,
                "classification_accuracy": 0.5,
            },
        ]
    ).to_csv(classifier_csv, index=False)

    output_dir = tmp_path / "out"
    assert (
        main(
            [
                "--source-calibration-summary-csv",
                str(source_csv),
                "--ranker-summary-csv",
                str(ranker_csv),
                "--reservoir-summary-csv",
                str(reservoir_csv),
                "--mixture-summary-csv",
                str(mixture_csv),
                "--viterbi-summary-csv",
                str(viterbi_csv),
                "--smoothing-summary-csv",
                str(smoothing_csv),
                "--classifier-summary-csv",
                str(classifier_csv),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    payload = json.loads(
        (output_dir / "mmuad_train_selected_config.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "raft-uav-mmuad-train-selected-config-v2"
    assert payload["source_calibration_mode"] == "source-translation"
    assert payload["source_translation_alpha"] == 0.5
    assert payload["point_extraction_mode"] == "static_dynamic_union"
    assert payload["ranker_model_type"] == "random-forest-classifier"
    assert payload["ranker_target_column"] == "good_cluster_5m"
    assert payload["candidate_reservoir_global_top_n"] == 40
    assert payload["candidate_reservoir_per_source_top_n"] == 3
    assert payload["candidate_reservoir_per_branch_top_n"] == 4
    assert payload["candidate_reservoir_max_candidates_per_frame"] == 60
    assert payload["candidate_reservoir_score_floor_quantile"] == 0.25
    assert payload["candidate_reservoir_cap_reason_bonus"] == 0.5
    assert payload["candidate_mixture_temperature"] == 64.0
    assert payload["candidate_mixture_sigma_log_weight"] == 2.0
    assert payload["candidate_mixture_huber_delta"] == 1.5
    assert payload["candidate_mixture_smoothness_weight"] == 3600.0
    assert payload["candidate_mixture_branch_balance"] == 0.25
    assert payload["candidate_mixture_source_balance"] == 0.5
    assert payload["mmuad_selection_mode"] == "viterbi"
    assert payload["viterbi_motion_weight"] == 2.0
    assert payload["viterbi_ranker_weight"] == 1.0
    assert payload["viterbi_source_switch_penalty"] == 0.25
    assert payload["viterbi_max_speed_mps"] == 40.0
    assert payload["smoothing_mode"] == "fixed-lag"
    assert payload["smoothing_speed_gate_mps"] == 20.0
    assert payload["smoothing_blend"] == 0.5
    assert payload["classifier_method"] == "random-forest"
    assert payload["image_nonimage_fusion_weight"] == 0.5
    assert (output_dir / "mmuad_train_selected_config_summary.csv").exists()


def test_train_selected_config_round_trip_preserves_new_fields(tmp_path: Path) -> None:
    config = validate_train_selected_config(
        {
            "point_extraction_mode": "static_dynamic_union",
            "candidate_reservoir_global_top_n": 40,
            "candidate_reservoir_per_source_top_n": 5,
            "candidate_mixture_temperature": 128,
            "candidate_mixture_smoothness_weight": 7200,
            "candidate_mixture_uniform_weight_floor": 0.05,
        }
    )
    output_json = tmp_path / "config.json"
    summary_csv = tmp_path / "summary.csv"

    payload = write_train_selected_config(
        config,
        output_json=output_json,
        summary_csv=summary_csv,
    )
    loaded = load_train_selected_config(output_json)

    assert payload["schema"] == "raft-uav-mmuad-train-selected-config-v2"
    assert loaded["point_extraction_mode"] == "static_dynamic_union"
    assert loaded["candidate_reservoir_global_top_n"] == 40
    assert loaded["candidate_reservoir_per_source_top_n"] == 5
    assert loaded["candidate_mixture_temperature"] == pytest.approx(128.0)
    assert loaded["candidate_mixture_uniform_weight_floor"] == pytest.approx(0.05)
    summary = pd.read_csv(summary_csv)
    assert int(summary.loc[0, "candidate_reservoir_global_top_n"]) == 40


def test_train_selected_config_validates_mixture_probability_fields() -> None:
    with pytest.raises(ValueError, match="candidate_mixture_branch_balance"):
        validate_train_selected_config({"candidate_mixture_branch_balance": 1.5})
    with pytest.raises(ValueError, match="candidate_reservoir_score_floor_quantile"):
        validate_train_selected_config({"candidate_reservoir_score_floor_quantile": -0.1})
