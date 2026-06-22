from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.cli import main as mmuad_cli_main
from raft_uav.mmuad.source_calibration import (
    apply_main,
    apply_source_calibration_json,
    apply_source_calibration_payload,
    fit_main,
    fit_source_calibration,
    write_source_calibration_json,
)


def _train_candidates() -> pd.DataFrame:
    truth = _train_truth()
    return pd.DataFrame(
        {
            "sequence_id": truth["sequence_id"],
            "time_s": truth["time_s"],
            "source": ["lidar_360", "lidar_360", "lidar_360"],
            "track_id": ["a", "b", "c"],
            "x_m": truth["x_m"] + 10.0,
            "y_m": truth["y_m"] - 4.0,
            "z_m": truth["z_m"] + 2.0,
            "confidence": 1.0,
        }
    )


def _train_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 2.0, 4.0],
            "y_m": [10.0, 11.0, 12.0],
            "z_m": [3.0, 3.5, 4.0],
        }
    )


def test_fit_source_translation_and_apply_corrects_candidates(tmp_path: Path) -> None:
    payload, pairs, fit_summary = fit_source_calibration(
        CandidateFrame(_train_candidates()),
        _train_truth(),
        mode="source-translation",
        max_truth_time_delta_s=0.1,
        max_pair_distance_m=50.0,
        min_pairs_per_source=2,
    )

    assert payload["protocol"] == "fit_on_train_only_apply_same_transform_to_val_or_test"
    assert len(pairs) == 3
    row = fit_summary.loc[fit_summary["source"] == "lidar_360"].iloc[0]
    assert row["fit_status"] == "fit"
    assert row["after_mean_m"] == pytest.approx(0.0)
    assert payload["transforms"]["lidar_360"]["translation_m"] == pytest.approx(
        [-10.0, 4.0, -2.0]
    )

    calibration_json = write_source_calibration_json(payload, tmp_path / "source_calibration.json")
    val_candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq999"],
                "time_s": [5.0],
                "source": ["lidar_360"],
                "track_id": ["v"],
                "x_m": [60.0],
                "y_m": [-1.0],
                "z_m": [9.0],
                "confidence": [1.0],
            }
        )
    )

    calibrated = apply_source_calibration_json(val_candidates, calibration_json)

    calibrated_row = calibrated.rows.iloc[0]
    assert calibrated_row["x_m"] == pytest.approx(50.0)
    assert calibrated_row["y_m"] == pytest.approx(3.0)
    assert calibrated_row["z_m"] == pytest.approx(7.0)
    assert bool(calibrated_row["mmuad_source_calibration_applied"]) is True
    assert calibrated_row["mmuad_source_calibration_mode"] == "source-translation"


def test_fit_source_translation_alpha_grid_uses_train_cv_shrinkage(tmp_path: Path) -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq_offset_a", "seq_offset_b", "seq_easy"],
            "time_s": [0.0, 0.0, 0.0],
            "x_m": [0.0, 10.0, 20.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "sequence_id": truth["sequence_id"],
            "time_s": truth["time_s"],
            "source": ["lidar_360", "lidar_360", "lidar_360"],
            "track_id": ["a", "b", "c"],
            "x_m": [10.0, 20.0, 20.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
            "confidence": [1.0, 1.0, 1.0],
        }
    )

    payload, _pairs, fit_summary = fit_source_calibration(
        CandidateFrame(candidates),
        truth,
        mode="source-translation",
        max_truth_time_delta_s=0.1,
        max_pair_distance_m=20.0,
        min_pairs_per_source=2,
        source_translation_alpha_grid=(0.0, 0.5, 1.0),
    )

    transform = payload["transforms"]["lidar_360"]
    row = fit_summary.loc[fit_summary["source"] == "lidar_360"].iloc[0]
    assert transform["source_translation_alpha"] == pytest.approx(0.5)
    assert transform["source_translation_base_translation_m"] == pytest.approx(
        [-10.0, 0.0, 0.0]
    )
    assert transform["translation_m"] == pytest.approx([-5.0, 0.0, 0.0])
    assert row["source_translation_alpha"] == pytest.approx(0.5)
    assert row["source_translation_alpha_cv_fold_count"] == 3

    calibration_json = write_source_calibration_json(payload, tmp_path / "source_calibration.json")
    calibrated = apply_source_calibration_json(CandidateFrame(candidates), calibration_json)
    assert set(calibrated.rows["mmuad_source_calibration_alpha"]) == {0.5}
    assert calibrated.rows.sort_values("track_id")["x_m"].tolist() == pytest.approx(
        [5.0, 15.0, 15.0]
    )


def test_fit_and_apply_source_calibration_commands_write_artifacts(tmp_path: Path) -> None:
    train_candidates = tmp_path / "train_candidates.csv"
    train_truth = tmp_path / "train_truth.csv"
    calibration_json = tmp_path / "mmuad_source_calibration.json"
    fit_pairs = tmp_path / "fit_pairs.csv"
    fit_summary = tmp_path / "fit_summary.csv"
    val_candidates = tmp_path / "val_candidates.csv"
    calibrated_candidates = tmp_path / "val_candidates_calibrated.csv"
    _train_candidates().to_csv(train_candidates, index=False)
    _train_truth().to_csv(train_truth, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seq002"],
            "time_s": [0.0],
            "source": ["lidar_360"],
            "track_id": ["v"],
            "x_m": [20.0],
            "y_m": [6.0],
            "z_m": [6.0],
            "confidence": [1.0],
        }
    ).to_csv(val_candidates, index=False)

    assert (
        fit_main(
            [
                "--train-candidates",
                str(train_candidates),
                "--train-truth",
                str(train_truth),
                "--mode",
                "source-translation",
                "--output-json",
                str(calibration_json),
                "--fit-pairs-csv",
                str(fit_pairs),
                "--fit-summary-csv",
                str(fit_summary),
                "--max-truth-time-delta-s",
                "0.1",
                "--max-pair-distance-m",
                "50",
                "--min-pairs-per-source",
                "2",
            ]
        )
        == 0
    )

    assert calibration_json.exists()
    assert fit_pairs.exists()
    assert fit_summary.exists()
    payload = json.loads(calibration_json.read_text(encoding="utf-8"))
    assert payload["provenance"]["train_candidates"] == str(train_candidates)

    assert (
        apply_main(
            [
                "--candidates",
                str(val_candidates),
                "--output-candidates",
                str(calibrated_candidates),
                "--mmuad-source-calibration-json",
                str(calibration_json),
            ]
        )
        == 0
    )

    output = pd.read_csv(calibrated_candidates).iloc[0]
    assert output["x_m"] == pytest.approx(10.0)
    assert output["y_m"] == pytest.approx(10.0)
    assert output["z_m"] == pytest.approx(4.0)


def test_mmuad_cli_applies_source_calibration_before_tracking(tmp_path: Path) -> None:
    payload, _pairs, _summary = fit_source_calibration(
        CandidateFrame(_train_candidates()),
        _train_truth(),
        mode="source-translation",
        max_truth_time_delta_s=0.1,
        max_pair_distance_m=50.0,
        min_pairs_per_source=2,
    )
    calibration_json = write_source_calibration_json(payload, tmp_path / "source_calibration.json")
    candidates_csv = tmp_path / "val_candidates.csv"
    calibrated_csv = tmp_path / "val_candidates_calibrated.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        {
            "sequence_id": ["seq002"],
            "time_s": [0.0],
            "source": ["lidar_360"],
            "track_id": ["v"],
            "x_m": [20.0],
            "y_m": [6.0],
            "z_m": [6.0],
            "confidence": [1.0],
        }
    ).to_csv(candidates_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seq002"],
            "time_s": [0.0],
            "x_m": [10.0],
            "y_m": [10.0],
            "z_m": [4.0],
        }
    ).to_csv(truth_csv, index=False)

    status = mmuad_cli_main(
        [
            "--candidate-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--mmuad-source-calibration-json",
            str(calibration_json),
            "--mmuad-source-calibrated-candidates-csv",
            str(calibrated_csv),
        ]
    )

    assert status == 0
    calibrated = pd.read_csv(calibrated_csv).iloc[0]
    assert calibrated["x_m"] == pytest.approx(10.0)
    assert calibrated["y_m"] == pytest.approx(10.0)
    assert calibrated["z_m"] == pytest.approx(4.0)
    estimates = pd.read_csv(output_dir / "mmuad_estimates.csv").iloc[0]
    assert estimates["state_x_m"] == pytest.approx(10.0)
    assert estimates["state_y_m"] == pytest.approx(10.0)
    assert estimates["state_z_m"] == pytest.approx(4.0)


def test_apply_source_calibration_rejects_explicit_mode_mismatch() -> None:
    payload, _pairs, _summary = fit_source_calibration(
        CandidateFrame(_train_candidates()),
        _train_truth(),
        mode="source-translation",
        max_truth_time_delta_s=0.1,
        max_pair_distance_m=50.0,
        min_pairs_per_source=2,
    )

    with pytest.raises(ValueError, match="does not match requested mode"):
        apply_source_calibration_payload(
            CandidateFrame(_train_candidates()),
            payload,
            mode="source-rigid",
        )
