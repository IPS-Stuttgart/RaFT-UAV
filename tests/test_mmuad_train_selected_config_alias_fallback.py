from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.train_selected_config import build_train_selected_config


def test_train_selected_config_falls_back_from_missing_preferred_aliases(
    tmp_path: Path,
) -> None:
    source_csv = tmp_path / "source.csv"
    classifier_csv = tmp_path / "classifier.csv"
    pd.DataFrame(
        [
            {
                "source_calibration_mode": None,
                "mode": "source-translation",
                "source_translation_alpha": None,
                "alpha": 0.25,
                "train_cv_pose_mse_loss_m2": 1.0,
            }
        ]
    ).to_csv(source_csv, index=False)
    pd.DataFrame(
        [
            {
                "classifier_method": None,
                "method": "hist-gradient-boosting",
                "image_nonimage_fusion_weight": None,
                "fusion_weight": None,
                "image_weight": 0.75,
                "classification_accuracy": 0.8,
            }
        ]
    ).to_csv(classifier_csv, index=False)

    config, records = build_train_selected_config(
        source_calibration_summary_csv=source_csv,
        classifier_summary_csv=classifier_csv,
    )

    assert config["source_calibration_mode"] == "source-translation"
    assert config["source_translation_alpha"] == 0.25
    assert config["classifier_method"] == "hist-gradient-boosting"
    assert config["image_nonimage_fusion_weight"] == 0.75
    assert records[0]["source_calibration_mode"] == "source-translation"
    assert records[1]["image_nonimage_fusion_weight"] == 0.75
