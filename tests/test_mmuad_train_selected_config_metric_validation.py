from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.train_selected_config import (
    build_train_selected_config,
    main,
)


def test_train_selected_config_rejects_summary_without_selection_metric(
    tmp_path: Path,
) -> None:
    summary_csv = tmp_path / "source_summary.csv"
    pd.DataFrame(
        [
            {"mode": "identity", "alpha": 0.0},
            {"mode": "source-translation", "alpha": 0.5},
        ]
    ).to_csv(summary_csv, index=False)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="no recognized selection metric column"):
        main(
            [
                "--source-calibration-summary-csv",
                str(summary_csv),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert not output_dir.exists()


def test_train_selected_config_rejects_summary_without_finite_metric(
    tmp_path: Path,
) -> None:
    summary_csv = tmp_path / "source_summary.csv"
    pd.DataFrame(
        [
            {
                "mode": "identity",
                "alpha": 0.0,
                "train_cv_pose_mse_loss_m2": "not-a-number",
            },
            {
                "mode": "source-translation",
                "alpha": 0.5,
                "train_cv_pose_mse_loss_m2": "inf",
            },
        ]
    ).to_csv(summary_csv, index=False)

    with pytest.raises(ValueError, match="no finite values"):
        build_train_selected_config(
            source_calibration_summary_csv=summary_csv,
        )


def test_train_selected_config_uses_later_recognized_finite_metric(
    tmp_path: Path,
) -> None:
    summary_csv = tmp_path / "source_summary.csv"
    pd.DataFrame(
        [
            {
                "mode": "identity",
                "alpha": 0.0,
                "train_cv_pose_mse_loss_m2": "not-a-number",
                "loso_pose_mse_loss_m2": 2.0,
            },
            {
                "mode": "source-translation",
                "alpha": 0.5,
                "train_cv_pose_mse_loss_m2": "not-a-number",
                "loso_pose_mse_loss_m2": 1.0,
            },
        ]
    ).to_csv(summary_csv, index=False)

    config, records = build_train_selected_config(
        source_calibration_summary_csv=summary_csv,
    )

    assert config["source_calibration_mode"] == "source-translation"
    assert config["source_translation_alpha"] == pytest.approx(0.5)
    assert records[0]["selection_metric"] == "loso_pose_mse_loss_m2"
    assert records[0]["selection_metric_value"] == pytest.approx(1.0)
