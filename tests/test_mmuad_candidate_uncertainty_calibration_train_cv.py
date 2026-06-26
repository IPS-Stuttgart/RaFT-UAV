from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_uncertainty_calibration import (
    load_candidate_sigma_calibration,
)
from raft_uav.mmuad.candidate_uncertainty_calibration_train_cv import (
    main as train_cv_main,
    select_candidate_sigma_calibration_by_sequence_cv,
)


def _feature_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sequence_id in ("seqA", "seqB", "seqC"):
        for time_s in range(4):
            records.extend(
                [
                    {
                        "sequence_id": sequence_id,
                        "time_s": float(time_s),
                        "source": "lidar_360",
                        "track_id": f"{sequence_id}-raw-{time_s}",
                        "candidate_branch": "raw",
                        "x_m": 2.0,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "confidence": 0.8,
                        "predicted_sigma_m": 1.0,
                        "truth_distance_3d_m": 2.0,
                    },
                    {
                        "sequence_id": sequence_id,
                        "time_s": float(time_s),
                        "source": "lidar_360",
                        "track_id": f"{sequence_id}-translated-{time_s}",
                        "candidate_branch": "translated",
                        "x_m": 6.0,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "confidence": 0.6,
                        "predicted_sigma_m": 1.0,
                        "truth_distance_3d_m": 6.0,
                    },
                ]
            )
    return pd.DataFrame.from_records(records)


def _candidate_rows() -> pd.DataFrame:
    return _feature_rows().drop(columns="truth_distance_3d_m")


def _truth_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sequence_id in ("seqA", "seqB", "seqC"):
        for time_s in range(4):
            records.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": float(time_s),
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_m": 0.0,
                }
            )
    return pd.DataFrame.from_records(records)


def test_train_cv_selects_lower_nll_global_quantile() -> None:
    calibration, folds, summary, selection = (
        select_candidate_sigma_calibration_by_sequence_cv(
            _feature_rows(),
            target_quantiles=(0.5, 0.8),
            min_group_rows_values=(100,),
            shrinkage_rows_values=(0.0,),
            scale_min=0.1,
            scale_max=10.0,
        )
    )

    assert len(folds) == 6
    assert len(summary) == 2
    assert summary.iloc[0]["selection_rank"] == 1
    assert selection["selected_config"]["target_quantile"] == 0.5
    assert calibration.target_quantile == 0.5
    assert summary.iloc[0]["gaussian_nll_3d"] < summary.iloc[1]["gaussian_nll_3d"]


def test_train_cv_refits_branch_specific_calibration_on_all_rows() -> None:
    calibration, folds, summary, selection = (
        select_candidate_sigma_calibration_by_sequence_cv(
            _feature_rows(),
            target_quantiles=(0.5,),
            min_group_rows_values=(1,),
            shrinkage_rows_values=(0.0,),
            scale_min=0.1,
            scale_max=10.0,
        )
    )

    raw_key = json.dumps(["lidar_360", "raw"], separators=(",", ":"))
    translated_key = json.dumps(["lidar_360", "translated"], separators=(",", ":"))
    assert calibration.source_branch_scales[raw_key] == pytest.approx(2.0)
    assert calibration.source_branch_scales[translated_key] == pytest.approx(6.0)
    assert len(folds) == 3
    assert len(summary) == 1
    assert selection["fold_count"] == 3
    assert selection["grid_size"] == 1


def test_candidate_sigma_calibration_train_cv_cli_writes_artifacts(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "train_cv"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = train_cv_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--target-quantile",
            "0.5",
            "--min-group-rows",
            "1",
            "--shrinkage-rows",
            "0",
            "--scale-min",
            "0.1",
            "--scale-max",
            "10",
            "--max-truth-time-delta-s",
            "0.01",
            "--write-features",
        ]
    )

    calibration_json = output_dir / "mmuad_candidate_sigma_calibration_train_selected.json"
    selection_json = output_dir / "mmuad_candidate_sigma_calibration_train_cv_selection.json"
    fold_csv = output_dir / "mmuad_candidate_sigma_calibration_train_cv_folds.csv"
    summary_csv = output_dir / "mmuad_candidate_sigma_calibration_train_cv_summary.csv"
    features_csv = output_dir / "mmuad_candidate_sigma_calibration_features.csv"

    assert status == 0
    assert calibration_json.exists()
    assert selection_json.exists()
    assert fold_csv.exists()
    assert summary_csv.exists()
    assert features_csv.exists()
    selection = json.loads(selection_json.read_text(encoding="utf-8"))
    assert selection["selected_config"] == {
        "target_quantile": 0.5,
        "min_group_rows": 1,
        "shrinkage_rows": 0.0,
    }
    assert len(pd.read_csv(fold_csv)) == 3
    assert pd.read_csv(summary_csv).loc[0, "selection_rank"] == 1
    calibration = load_candidate_sigma_calibration(calibration_json)
    assert calibration.calibration_row_count == len(_feature_rows())
