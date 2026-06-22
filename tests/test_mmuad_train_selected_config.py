from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.train_selected_config import main


def test_train_selected_config_writes_required_fields_from_train_summaries(
    tmp_path: Path,
) -> None:
    source_csv = tmp_path / "source.csv"
    ranker_csv = tmp_path / "ranker.csv"
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
                "loso_pose_mse_loss_m2": 15.0,
            },
        ]
    ).to_csv(ranker_csv, index=False)
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
    assert payload["schema"] == "raft-uav-mmuad-train-selected-config-v1"
    assert payload["source_calibration_mode"] == "source-translation"
    assert payload["source_translation_alpha"] == 0.5
    assert payload["ranker_model_type"] == "random-forest-classifier"
    assert payload["ranker_target_column"] == "good_cluster_5m"
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
